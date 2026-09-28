import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.task3.data import make_demo
from src.task4.model import RetrievalModel, contrastive_loss
from src.task4.metrics import retrieval_metrics, recompute
from src.task4.train import train, evaluate


class RetrievalTests(unittest.TestCase):
    def test_known_ranks_and_ties(self):
        identity = np.eye(12, dtype=np.float32)
        perfect = retrieval_metrics(identity, identity, query_batch_size=3)
        self.assertEqual(perfect["mean_recall_at_1"], 1)
        self.assertEqual(perfect["text_to_graph"]["mean_reciprocal_rank"], 1)
        self.assertAlmostEqual(perfect["random_ranking_expected_recall"]["1"], 1 / 12)
        shifted = retrieval_metrics(identity, np.roll(identity, 1, axis=0))
        self.assertEqual(shifted["mean_recall_at_1"], 0)
        ties = retrieval_metrics(np.ones((12, 3)), np.ones((12, 3)))
        self.assertEqual(ties["text_to_graph"]["recall_at_10"], 0)
        self.assertAlmostEqual(ties["text_to_graph"]["mean_reciprocal_rank"], 1 / 12)
        self.assertEqual(perfect, retrieval_metrics(identity, identity, query_batch_size=1))

    def test_loss_and_gradients(self):
        model = RetrievalModel(4, 5, 3)
        t, g = model(torch.randn(4, 4), torch.randn(4, 5))
        loss = contrastive_loss(t, g)
        loss.backward()
        self.assertGreater(model.text.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.graph.weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(t.norm(dim=1), torch.ones(4))
        with self.assertRaises(ValueError):
            contrastive_loss(t[:1], g[:1])
        with self.assertRaises(ValueError):
            contrastive_loss(t, g, 0)

    def test_training_reload_and_feature_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = make_demo(root / "features.pt")
            with contextlib.redirect_stdout(io.StringIO()):
                report = train(features, root / "run", epochs=2, batch_size=17, device="cpu")
            self.assertEqual(report["split"], "validation")
            self.assertEqual(recompute(root / "run/validation/embeddings.npz"), report["metrics"])
            self.assertEqual(recompute(root / "run/validation/untrained_embeddings.npz"), report["untrained_projection_metrics"])
            self.assertAlmostEqual(report["mean_recall_at_1_gain"], report["metrics"]["mean_recall_at_1"] - report["untrained_projection_metrics"]["mean_recall_at_1"])
            again = evaluate(root / "run/best_model.pt", features, root / "eval", split="validation", device="cpu")
            self.assertEqual(report, again)
            test = evaluate(root / "run/best_model.pt", features, root / "test", device="cpu")
            self.assertEqual(test["metrics"]["gallery_size"], 6)
            self.assertTrue((root / "test/examples.json").exists())
            make_demo(features, seed=99)
            with self.assertRaisesRegex(ValueError, "differs"):
                evaluate(root / "run/best_model.pt", features, root / "bad")

    def test_recompute_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(path, text=np.eye(2), graph=np.eye(2), sample_ids=["same", "same"])
            with self.assertRaisesRegex(ValueError, "unique"):
                recompute(path)
