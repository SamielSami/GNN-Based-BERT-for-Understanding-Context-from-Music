import contextlib
import io
import tempfile
import json
import unittest
from pathlib import Path

import torch

from src.task3.data import load_features, make_demo
from src.task3.models import FusionClassifier, MODES
from src.task3.train import run_experiments
from src.task3.align import align_text
from src.task3.prepare import prepare_features
from src.task3.evaluate import evaluate_checkpoint
import numpy as np


class FusionTests(unittest.TestCase):
    def test_alignment_and_mismatched_encoder_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [dict(sample_id=str(i), text=f"caption {i}", labels=[1, 0], split=s)
                       for i, s in enumerate(("train", "validation", "test"))]
            manifest = dict(label_names=["a", "b"], records=records)
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            align_text(path, root / "text")
            splits = json.loads((root / "text/split_indices.json").read_text())
            self.assertEqual(splits, {"train": [0], "validation": [1], "test": [2]})
            torch.save(dict(config={}, tags=["b", "a"]), root / "text.pt")
            torch.save(dict(config={}, label_names=["a", "b"]), root / "graph.pt")
            with self.assertRaisesRegex(ValueError, "Checkpoint label"):
                prepare_features(path, path, root / "text.pt", root / "graph.pt", root / "out.pt")

    def test_shapes_gradients_and_masks(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                model = FusionClassifier(12, 8, 3, mode, 16).eval()
                tokens = torch.randn(2, 5, 12, requires_grad=True)
                graph = torch.randn(2, 8, requires_grad=True)
                mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
                output = model(tokens, mask, graph)
                self.assertEqual(output.shape, (2, 3))
                changed = tokens.detach().clone()
                changed[~mask.bool()] = 999
                torch.testing.assert_close(output, model(changed, mask, graph))
                changed[~mask.bool()] = float("nan")
                torch.testing.assert_close(output, model(changed, mask, graph))
                output.sum().backward()
                if mode != "gnn":
                    self.assertGreater(tokens.grad.abs().sum().item(), 0)
                    self.assertEqual(tokens.grad[~mask.bool()].abs().sum().item(), 0)
                if mode != "bert":
                    self.assertGreater(graph.grad.abs().sum().item(), 0)
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if mode == "cross_attention":
                    self.assertGreater(tokens.grad[:, 1:3].abs().sum().item(), 0)
                torch.testing.assert_close(output[:1], model(tokens[:1], mask[:1], graph[:1]))
                with self.assertRaises(ValueError):
                    model(tokens, torch.zeros_like(mask), graph)

    def test_all_ablation_runs_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = make_demo(root / "features.pt")
            with contextlib.redirect_stdout(io.StringIO()):
                report = run_experiments(features, root / "results", epochs=1, hidden_dim=8, device="cpu")
            self.assertEqual(report["evaluated_split"], "validation")
            self.assertTrue(report["provenance"]["synthetic"])
            for mode in MODES:
                self.assertTrue((root / "results" / f"{mode}.pt").exists())
            for filename in ("comparison.csv", "cases.json", "embeddings.png", "selected_predictions.npz",
                             "training_curves.png", "validation_curves.png"):
                self.assertTrue((root / "results" / filename).exists())
            selection = json.loads((root / "results/selection.json").read_text())
            checkpoint = root / "results" / selection["checkpoint"]
            evaluated = evaluate_checkpoint(checkpoint, features, root / "evaluation", split="validation", device="cpu")
            self.assertEqual(evaluated["metrics"], report["metrics"])
            with np.load(root / "results/selected_predictions.npz") as original, np.load(root / "evaluation/predictions.npz") as restored:
                np.testing.assert_allclose(original["probabilities"], restored["probabilities"], atol=1e-6)
                np.testing.assert_array_equal(original["sample_ids"], restored["sample_ids"])
            data = load_features(features)
            data["tokens"][0, 0, 0] += 1
            torch.save(data, features)
            with self.assertRaisesRegex(ValueError, "Feature file differs"):
                evaluate_checkpoint(checkpoint, features, root / "bad_eval", device="cpu")
            data["sample_ids"][1] = data["sample_ids"][0]
            torch.save(data, features)
            with self.assertRaises(ValueError):
                load_features(features)

    def test_early_stopping_and_invalid_learning_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = make_demo(root / "features.pt")
            with self.assertRaises(ValueError):
                run_experiments(features, root / "invalid", learning_rate=float("nan"))
            with contextlib.redirect_stdout(io.StringIO()):
                run_experiments(features, root / "results", epochs=8, patience=1,
                                learning_rate=1e-30, hidden_dim=8, device="cpu")
            for mode in MODES:
                history = json.loads((root / "results" / f"{mode}_history.json").read_text())
                self.assertEqual(len(history), 2)
