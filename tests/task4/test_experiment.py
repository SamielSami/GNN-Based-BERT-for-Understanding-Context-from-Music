import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.task3.data import make_demo
from src.task4.experiment import run_experiment, evaluate_suite, verify, pairing_audit
from src.task4.model import RetrievalModel, contrastive_loss
from src.task4.train import paired_inputs


class RetrievalSuiteTests(unittest.TestCase):
    def test_suite_freezes_validation_then_audits_test_and_detects_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = make_demo(root / "features.pt")
            run = root / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_experiment(features, run, seeds=(42, 43), epochs=2, batch_size=17)
            self.assertFalse(result["evaluated_test"])
            self.assertFalse((run / "seed42/test").exists())
            selection = (run / "selection.json").read_bytes()
            result = evaluate_suite(run)
            self.assertTrue(result["valid"])
            self.assertEqual(selection, (run / "selection.json").read_bytes())
            self.assertEqual(result, verify(run))
            with self.assertRaisesRegex(ValueError, "already exist"):
                evaluate_suite(run)
            with self.assertRaisesRegex(ValueError, "empty output"):
                run_experiment(features, run, epochs=1)
            path = run / "seed42/test/metrics.json"
            original = path.read_bytes()
            value = json.loads(original)
            value["metrics"]["text_to_graph"]["recall_at_1"] = 0.123
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Test artifact changed"):
                verify(run)
            path.write_bytes(original)
            checkpoint = run / "seed42/best_model.pt"
            with checkpoint.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                verify(run)

    def test_caption_and_video_audit(self):
        data = dict(sample_ids=["video1_0_10", "video1_10_20", "video2_0_10"],
                    texts=["LOUD music!", " loud MUSIC ", "other"], splits=["train", "train", "test"])
        audit = pairing_audit(data)
        self.assertEqual(len(audit["normalized_caption_duplicates"]), 1)
        self.assertEqual(len(audit["exact_caption_duplicates"]), 0)
        self.assertEqual(len(audit["repeated_videos"]), 1)
        data["splits"][1] = "validation"
        with self.assertRaisesRegex(ValueError, "overlap"):
            pairing_audit(data)

    def test_cached_inputs_are_detached_and_loss_ignores_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = make_demo(Path(directory) / "features.pt")
            data = torch.load(path, weights_only=True)
            data["tokens"].requires_grad_(True)
            data["graph"].requires_grad_(True)
            torch.save(data, path)
            _, text, graph = paired_inputs(path)
            model = RetrievalModel(text.shape[1], graph.shape[1], 4)
            before = text.clone(), graph.clone()
            loss = contrastive_loss(*model(text[:8], graph[:8]))
            loss.backward()
            self.assertFalse(text.requires_grad)
            self.assertFalse(graph.requires_grad)
            torch.testing.assert_close(text, before[0])
            torch.testing.assert_close(graph, before[1])
            data["labels"] = 1 - data["labels"]
            other_path = Path(directory) / "changed_labels.pt"
            torch.save(data, other_path)
            _, other_text, other_graph = paired_inputs(other_path)
            torch.testing.assert_close(loss, contrastive_loss(*model(other_text[:8], other_graph[:8])))

    def test_asymmetric_known_ranks(self):
        from src.task4.metrics import retrieval_metrics
        # graph 0 is top for both texts; text 1 is top for both graphs.
        text = np.array([[1., 0.], [0.8, 0.6]])
        graph = np.array([[0.8, 0.6], [0., 1.]])
        metrics = retrieval_metrics(text, graph)
        self.assertEqual(metrics["text_to_graph"]["recall_at_1"], 0.5)
        self.assertEqual(metrics["graph_to_text"]["recall_at_1"], 0.5)
        self.assertEqual(metrics["text_to_graph"]["mean_reciprocal_rank"], 0.75)
