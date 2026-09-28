"""Cache and evaluation regressions using small offline tensors."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.task2.evaluate import evaluate_checkpoint
from src.task2.features import AudioFeatureConfig
from src.task2.preprocess import preprocess_manifest
from src.task2.train import Task2TrainConfig


class CacheSafetyTests(unittest.TestCase):
    def test_cache_refreshes_labels_splits_and_changed_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.bin"
            audio.write_bytes(b"original")
            path = root / "manifest.json"
            manifest = {"label_names": ["a", "b"], "records": [
                {"sample_id": "a", "audio_path": str(audio), "split": "train", "labels": [1, 0]},
                {"sample_id": "b", "audio_path": str(audio), "split": "test", "labels": [0, 1]},
            ]}
            config = AudioFeatureConfig()

            def extract(record, manifest_path, feature_config):
                value = 1.0 if record["sample_id"] == "a" else 7.0
                return {"sample_id": record["sample_id"], "audio_path": str(audio),
                        "x_raw": torch.full((2, 3), value), "mel": torch.ones(1, 4, 4),
                        "feature_config": feature_config.to_dict()}

            with patch("src.task2.preprocess.extract_raw_sample", side_effect=extract) as mocked, contextlib.redirect_stdout(io.StringIO()):
                path.write_text(json.dumps(manifest), encoding="utf-8")
                preprocess_manifest(path, root / "processed", config=config)
                self.assertEqual(mocked.call_count, 2)
                manifest["records"][0]["split"] = "test"
                manifest["records"][1]["split"] = "train"
                manifest["records"][1]["labels"] = [1, 1]
                path.write_text(json.dumps(manifest), encoding="utf-8")
                processed = preprocess_manifest(path, root / "processed", config=config)
                self.assertEqual(mocked.call_count, 2)  # Features reused; metadata refreshed.
                norm = json.loads((processed.parent / "normalization.json").read_text())
                self.assertEqual(norm["mean"], [7.0, 7.0, 7.0])
                records = json.loads(processed.read_text())["records"]
                sample = torch.load(processed.parent / records[1]["processed_path"], weights_only=False)
                self.assertEqual(sample["labels"].tolist(), [1, 1])
                audio.write_bytes(b"modified")
                preprocess_manifest(path, root / "processed", config=config)
                self.assertEqual(mocked.call_count, 4)

    def test_reordered_labels_rejected_before_model_loading(self):
        checkpoint = {"config": {"graph_variant": "temporal"}, "label_names": ["a", "b"]}
        with patch("src.task2.evaluate.torch.load", return_value=checkpoint), patch(
            "src.task2.evaluate.ProcessedTask2Dataset"
        ) as dataset:
            dataset.return_value.label_names = ["b", "a"]
            with self.assertRaisesRegex(ValueError, "label names/order"):
                evaluate_checkpoint("unused.pt", "unused.json", device_name="cpu")

    def test_invalid_training_settings_rejected(self):
        for kwargs in ({"epochs": 0}, {"batch_size": 0}, {"dropout": 1}, {"num_workers": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Task2TrainConfig(manifest="unused.json", **kwargs)
