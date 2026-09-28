"""Offline leakage, ID alignment, calibration and inference regressions."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.task1.audioset import (build_audioset_frame, load_prepared_manifest,
                               parse_labels, save_audioset_dataset, split_by_id)
from src.task1.baselines import fit_baselines
from src.task1.predict import load_checkpoint, predict
from src.task1.train import Task1Config, load_tag_data, select_thresholds, train


def fixture():
    return pd.DataFrame({"ytid": [f"video_{i:05}" for i in range(40)],
                         "caption": [f"A unique music description number {i}" for i in range(40)],
                         "audioset_positive_labels": ["/m/guitar" if i % 2 else "/m/piano" for i in range(40)]})


class AcademicDataTests(unittest.TestCase):
    def test_id_split_is_order_invariant_and_groups_repeated_ids(self):
        raw = fixture()
        raw = pd.concat([raw, raw.iloc[[0]].assign(caption="A second window of the same video")], ignore_index=True)
        _, manifest = build_audioset_frame(raw, min_train_support=1)
        _, reordered = build_audioset_frame(raw.iloc[::-1], min_train_support=1)
        self.assertEqual(manifest["split_ids"], reordered["split_ids"])
        owners = [name for name, rows in manifest["split_indices"].items() if 0 in rows or 40 in rows]
        self.assertEqual(len(owners), 1)

    def test_heldout_labels_cannot_select_vocabulary_or_remove_rows(self):
        raw = fixture()
        first, manifest = build_audioset_frame(raw, top_k=1, min_train_support=1)
        held_ids = manifest["split_ids"]["validation"] + manifest["split_ids"]["test"]
        raw.loc[raw.ytid.isin(held_ids), "audioset_positive_labels"] = "/m/heldout_only,/m/new"
        second, changed = build_audioset_frame(raw, top_k=1, min_train_support=1)
        self.assertEqual(manifest["tags"], changed["tags"])
        self.assertEqual(manifest["train_support"], changed["train_support"])
        self.assertEqual(manifest["split_ids"], changed["split_ids"])
        self.assertEqual(len(second), len(raw))
        self.assertEqual(second.loc[second.ytid.isin(held_ids), changed["tags"]].to_numpy().sum(), 0)
        pd.testing.assert_frame_equal(first.iloc[manifest["split_indices"]["train"]], second.iloc[manifest["split_indices"]["train"]])

    def test_caption_and_aspect_changes_do_not_create_labels(self):
        raw = fixture()
        frame, manifest = build_audioset_frame(raw, min_train_support=1)
        raw.caption = [f"no guitar piano singing invented text {i}" for i in range(len(raw))]
        raw["aspect_list"] = "['secret target', 'rock', 'piano']"
        other, changed = build_audioset_frame(raw, min_train_support=1)
        self.assertEqual(manifest["tags"], changed["tags"])
        np.testing.assert_array_equal(frame[manifest["tags"]], other[changed["tags"]])
        self.assertNotIn("aspect_list", other.columns)

    def test_invalid_fields_and_train_constant_targets(self):
        for column in ("ytid", "caption", "audioset_positive_labels"):
            with self.subTest(column=column), self.assertRaises(ValueError):
                build_audioset_frame(fixture().drop(columns=column), min_train_support=1)
        with self.assertRaises(ValueError):
            parse_labels("guitar, happy")
        raw = fixture()
        raw.audioset_positive_labels += ",/m/constant"
        _, manifest = build_audioset_frame(raw, min_train_support=1)
        self.assertNotIn("/m/constant", manifest["tags"])
        self.assertIn("/m/constant", manifest["excluded_train_constant_labels"])

    def test_manifest_hash_id_and_support_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.csv"
            frame, manifest = save_audioset_dataset(fixture(), path, min_train_support=1)
            self.assertEqual(load_prepared_manifest(path), manifest)
            texts, labels, tags = load_tag_data(path)
            self.assertNotIn("ytid", tags)
            self.assertEqual(len(texts), len(frame))
            with self.assertRaises(FileExistsError):
                save_audioset_dataset(fixture(), path, min_train_support=1)
            manifest_path = path.with_suffix(".manifest.json")
            manifest["split_ids"]["test"].append(manifest["split_ids"]["train"][0])
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ytid"):
                load_prepared_manifest(path)
            frame.iloc[0, 1] = "tampered caption"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "hash"):
                load_prepared_manifest(path)

    def test_explicit_resplit_of_fitted_vocabulary_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.csv"
            _, manifest = save_audioset_dataset(fixture(), path, min_train_support=1)
            splits = manifest["split_indices"]
            splits["train"][0], splits["test"][0] = splits["test"][0], splits["train"][0]
            split_path = Path(directory) / "wrong_splits.json"
            split_path.write_text(json.dumps(splits), encoding="utf-8")
            with patch("src.task1.train.AutoModel.from_pretrained") as model:
                with self.assertRaisesRegex(ValueError, "partition"):
                    train(Task1Config(data_path=str(path), split_path=str(split_path), output_dir=str(Path(directory) / "run")))
                model.assert_not_called()


class AcademicCalibrationTests(unittest.TestCase):
    def test_thresholds_use_validation_and_handle_absent_labels(self):
        targets = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
        probabilities = np.array([[0.3, 0.1], [0.1, 0.2], [0.4, 0.1], [0.2, 0.1]])
        calibration = select_thresholds(targets, probabilities)
        self.assertEqual(calibration["fit_split"], "validation")
        self.assertAlmostEqual(calibration["thresholds"][0], 0.3)
        self.assertEqual(calibration["no_validation_positives"], [1])
        self.assertEqual(calibration["thresholds"][1], calibration["global_threshold"])
        with self.assertRaises(ValueError):
            select_thresholds(targets, probabilities * np.nan)

    def test_tfidf_vocabulary_fitted_on_train_and_not_heldout(self):
        raw = fixture()
        frame, manifest = build_audioset_frame(raw, min_train_support=1)
        splits = manifest["split_indices"]
        for i in splits["validation"] + splits["test"]:
            frame.loc[i, "text"] += " heldouttoken"
        models = fit_baselines(frame.text.tolist(), frame[manifest["tags"]].to_numpy(), manifest["tags"], splits, 42)
        self.assertNotIn("heldouttoken", models["tfidf_logistic"][0][0].vocabulary_)
        self.assertTrue(all(cal["fit_split"] == "validation" for _, cal in models.values()))

    def test_checkpoint_and_inference_use_per_label_thresholds_without_top_k_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"state_dict": {}, "config": {"model_name": "offline", "max_length": 8, "threshold": 0.5},
                        "tags": ["low", "high"], "thresholds": [0.2, 0.95]}, path)
            _, metadata = load_checkpoint(path)
            self.assertEqual(metadata["thresholds"], [0.2, 0.95])
            class FakeModel(torch.nn.Module):
                def forward(self, **kwargs):
                    return torch.logit(torch.tensor([[0.3, 0.9]]))
            tokenizer = lambda *args, **kwargs: {"input_ids": torch.ones((1, 2), dtype=torch.long)}
            with patch("src.task1.predict.AutoTokenizer.from_pretrained", return_value=tokenizer), patch("src.task1.predict.BertTagClassifier", return_value=FakeModel()):
                result = predict(["text"], path, top_k=1)[0]
            self.assertEqual([item["tag"] for item in result["predicted_tags"]], ["low"])
            self.assertEqual([item["tag"] for item in result["top_tags"]], ["high"])


if __name__ == "__main__":
    unittest.main()
