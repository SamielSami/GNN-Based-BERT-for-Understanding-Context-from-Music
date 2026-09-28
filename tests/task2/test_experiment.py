"""Checks for identical cohorts, frozen test selection and cache-bound checkpoints."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.task2.dataset import dataset_identity, load_manifest
from src.task2.experiment import run_experiment, evaluate_experiment
from src.task2.graphs import build_edge_variants
from src.task2.train import Task2TrainConfig, classification_metrics, select_global_threshold


class ExperimentTests(unittest.TestCase):
    def test_frozen_suite_predictions_and_changed_cache_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(321)
            records = []
            for i, split in enumerate(["train"] * 6 + ["validation"] * 3 + ["test"] * 3):
                x = rng.normal(size=(6, 8)).astype(np.float32)
                record = {"sample_id": str(i), "split": split, "processed_path": f"sample{i}.pt"}
                torch.save({**record, "x": torch.from_numpy(x), "mel": torch.from_numpy(rng.random((1, 16, 16)).astype(np.float32)),
                            "labels": torch.tensor([i % 2, int(i % 3 == 0)], dtype=torch.float32),
                            "edge_indices": build_edge_variants(x, top_k=1, seed=i)}, root / record["processed_path"])
                records.append(record)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"label_names": ["a", "b"], "synthetic": True, "records": records}))
            output = root / "runs"
            config = Task2TrainConfig(manifest=str(manifest), output_dir=str(output), epochs=1, patience=1,
                                      hidden_dim=8, layers=1, batch_size=3, device="cpu", cpu_threads=2)
            with contextlib.redirect_stdout(io.StringIO()):
                report = run_experiment(config)
            self.assertEqual(len(report["rows"]), 5)
            self.assertTrue(report["synthetic"])
            self.assertFalse(report["evaluated_test"])
            selection_bytes = (output / "selection.json").read_bytes()
            original_fingerprint = dataset_identity(manifest)["sha256"]
            sample_path = root / records[0]["processed_path"]
            original_bytes = sample_path.read_bytes()
            sample = torch.load(sample_path, weights_only=False)
            sample["x"][0, 0] += .1
            torch.save(sample, sample_path)
            self.assertNotEqual(dataset_identity(manifest)["sha256"], original_fingerprint)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                evaluate_experiment(output, device="cpu")
            sample_path.write_bytes(original_bytes)
            report = evaluate_experiment(output, device="cpu")
            self.assertTrue(report["evaluated_test"])
            self.assertEqual(selection_bytes, (output / "selection.json").read_bytes())
            for row, run in zip(report["rows"], report["selection"]["runs"]):
                path = output / run["directory"]
                with np.load(path / "validation_predictions.npz") as prediction:
                    threshold = select_global_threshold(prediction["targets"], prediction["probabilities"])
                with np.load(path / "test_predictions.npz") as prediction:
                    self.assertEqual(prediction["sample_ids"].tolist(), ["9", "10", "11"])
                    self.assertAlmostEqual(float(prediction["threshold"]), threshold)
                    metrics = classification_metrics(prediction["targets"], prediction["probabilities"], ["a", "b"], threshold)
                    self.assertAlmostEqual(metrics["macro_f1"], row["test_macro_f1"])
                    self.assertAlmostEqual(metrics["micro_f1"], row["test_micro_f1"])
                    self.assertAlmostEqual(metrics["mean_average_precision"], row["test_map"])
            self.assertTrue((output / "graph_ablation.csv").is_file())
            with self.assertRaises(FileExistsError):
                run_experiment(config)
            with self.assertRaises(FileExistsError):
                evaluate_experiment(output, device="cpu")

    def test_invalid_manifest_and_metric_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for record in ({"sample_id": "x", "split": "typo"}, {"sample_id": "x", "split": "train", "labels": [float("nan")]}):
                path.write_text(json.dumps({"label_names": ["a"], "records": [record]}))
                with self.assertRaises(ValueError):
                    load_manifest(path)
        with self.assertRaisesRegex(ValueError, "finite probabilities"):
            classification_metrics(np.array([[1, 0]]), np.array([[.5, np.nan]]), ["a", "b"], .5)


if __name__ == "__main__":
    unittest.main()
