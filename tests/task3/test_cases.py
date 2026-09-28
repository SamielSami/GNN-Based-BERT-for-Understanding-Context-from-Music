import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch_geometric.data import Data

from src.task2.models import build_task2_model
from src.task3.cases import generate_cases, label_outcomes, representation_sensitivity, select_case_indices
from src.task3.data import feature_fingerprint
from src.task3.models import FusionClassifier


class MeanEncoder(nn.Module):
    def forward(self, graph):
        return graph.x.mean(0, keepdim=True)


class CaseEvidenceTests(unittest.TestCase):
    def test_fixed_validation_selection_and_named_error_types(self):
        self.assertEqual(select_case_indices(["test", "validation", "train", "validation", "validation", "validation"]), [1, 3, 4])
        with self.assertRaisesRegex(ValueError, "At least 3"):
            select_case_indices(["validation"])
        rows = label_outcomes([0.8, 0.7, 0.2, 0.1], [1, 0, 1, 0], ["a", "b", "c", "d"], {"a": "Music"}, 0.5)
        self.assertEqual([row["status"] for row in rows], ["true_positive", "false_positive", "false_negative", "true_negative"])
        self.assertEqual(rows[0]["name"], "Music")

    def test_signed_graph_effects_cls_limit_and_inputs_preserved(self):
        model = FusionClassifier(2, 2, 2, "concat", 2)
        with torch.no_grad():
            model.text_projection.weight.copy_(torch.eye(2)); model.text_projection.bias.zero_()
            model.graph_projection.weight.copy_(torch.eye(2)); model.graph_projection.bias.zero_()
            model.head[1].weight.copy_(torch.tensor([[1.0, 0, 1, 0], [-1.0, 0, -1, 0]]))
            model.head[1].bias.zero_()
        graph = Data(x=torch.tensor([[1.0, 0], [3.0, 2]]), edge_index=torch.tensor([[0, 1], [1, 0]]))
        tokens = torch.tensor([[[1.0, 2], [4.0, 5], [2.0, 3], [999.0, 999]]])
        mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
        encoder = MeanEncoder()
        original_tokens, original_graph = tokens.clone(), graph.clone()
        result = representation_sensitivity(model, encoder, tokens, mask, encoder(graph), graph,
                                            ["[CLS]", "guitar", "[SEP]", "[PAD]"])
        self.assertEqual(result["explained_label_index"], 0)
        self.assertAlmostEqual(result["probabilities"][0], torch.sigmoid(torch.tensor(3.0)).item())
        expected_delta = (torch.sigmoid(torch.tensor(3.0)) - torch.sigmoid(torch.tensor(2.5))).item()
        self.assertAlmostEqual(result["segments"][0]["probability_delta"], expected_delta)
        self.assertEqual(result["segments"][1]["start_seconds"], 1.0)
        self.assertEqual(len(result["tokens"]), 3)  # Padding never receives attribution.
        self.assertEqual(result["tokens"][1]["probability_delta"], 0)
        self.assertGreater(result["tokens"][0]["probability_delta"], 0)
        other_label = representation_sensitivity(model, encoder, tokens, mask, encoder(graph), graph,
                                                ["[CLS]", "guitar", "[SEP]", "[PAD]"], excluded_label_indices=[0])
        self.assertEqual(other_label["explained_label_index"], 1)
        self.assertLess(other_label["segments"][0]["probability_delta"], 0)
        self.assertEqual(len(other_label["segments"][0]["probability_deltas"]), 2)
        torch.testing.assert_close(tokens, original_tokens)
        torch.testing.assert_close(graph.x, original_graph.x)
        torch.testing.assert_close(graph.edge_index, original_graph.edge_index)
        with self.assertRaisesRegex(ValueError, "Recomputed graph embedding"):
            representation_sensitivity(model, encoder, tokens, mask, encoder(graph) + 0.1, graph,
                                       ["[CLS]", "guitar", "[SEP]", "[PAD]"])

    def test_cross_attention_sees_lexical_vectors_and_graph_only_does_not(self):
        torch.manual_seed(2)
        graph = Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))
        encoder, tokens, mask = MeanEncoder(), torch.randn(1, 3, 2), torch.ones(1, 3, dtype=torch.bool)
        cross = representation_sensitivity(FusionClassifier(2, 2, 2, "cross_attention", 4), encoder,
                                           tokens, mask, encoder(graph), graph, ["[CLS]", "guitar", "[SEP]"])
        self.assertGreater(abs(cross["tokens"][1]["probability_delta"]), 1e-8)
        gnn = representation_sensitivity(FusionClassifier(2, 2, 2, "gnn", 4), encoder,
                                         tokens, mask, encoder(graph), graph, ["[CLS]", "guitar", "[SEP]"])
        self.assertTrue(all(row["probability_delta"] == 0 for row in gnn["tokens"]))

    def test_artifacts_restore_encoder_and_reject_changed_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(7)
            labels = ["/m/04rlf", "/m/0342h"]
            ids = ["train", "validation_z", "validation_a", "validation_y", "test"]
            splits = ["train", "validation", "validation", "validation", "test"]
            graph_model = build_task2_model("gnn", input_dim=2, num_labels=2, hidden_dim=4, layers=1, dropout=0).eval()
            graph_checkpoint = root / "graph.pt"
            torch.save(dict(state_dict=graph_model.state_dict(), input_dim=2, label_names=labels,
                            config=dict(model="gnn", graph_variant="temporal", hidden_dim=4, layers=1,
                                        dropout=0, manifest="Z:/stale/manifest.json")), graph_checkpoint)
            rows, embeddings = [], []
            targets = torch.tensor([[1.0, 0], [1.0, 1], [0.0, 1], [1.0, 0], [1.0, 1]])
            for index, sample_id in enumerate(ids):
                graph = Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))
                with torch.no_grad():
                    embeddings.append(graph_model.encoder(graph)[0])
                filename = f"sample_{index}.pt"
                torch.save(dict(sample_id=sample_id, split=splits[index], x=graph.x,
                                mel=torch.ones(1, 4, 8), labels=targets[index], edge_indices={"temporal": graph.edge_index}), root / filename)
                rows.append(dict(sample_id=sample_id, split=splits[index], labels=targets[index].tolist(),
                                 processed_path=filename, start_s=20, end_s=23))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(label_names=labels, records=rows,
                                                feature_config=dict(segment_seconds=1.0))), encoding="utf-8")
            features = root / "features.pt"
            torch.save(dict(sample_ids=ids, splits=splits, label_names=labels, tokens=torch.randn(5, 4, 3),
                            mask=torch.tensor([[1, 1, 1, 0]] * 5), graph=torch.stack(embeddings), labels=targets,
                            texts=["Guitar music."] * 5, token_strings=[["[CLS]", "guitar", "[SEP]", "[PAD]"]] * 5,
                            provenance=dict(synthetic=True, graph_checkpoint_sha256=feature_fingerprint(graph_checkpoint),
                                            processed_manifest_sha256=feature_fingerprint(manifest))), features)
            dimensions = dict(text_dim=3, graph_dim=8, num_labels=2, hidden_dim=4)
            model = FusionClassifier(**dimensions, mode="cross_attention")
            checkpoint = root / "fusion.pt"
            torch.save(dict(state_dict=model.state_dict(), mode="cross_attention", dimensions=dimensions,
                            threshold=0.5, label_names=labels, feature_sha256=feature_fingerprint(features)), checkpoint)
            report = generate_cases(checkpoint, features, manifest, graph_checkpoint, root / "cases")
            self.assertEqual([case["sample_id"] for case in report["cases"]], ids[1:4])
            self.assertTrue(report["provenance"]["synthetic"])
            self.assertIn("neither causal", report["interpretation"])
            self.assertEqual(report["cases"][0]["label_outcomes"][0]["name"], "Music")
            self.assertEqual(report["cases"][0]["explained_label"]["name"], "Guitar")
            for filename in ("cases.json", "cases.md", "case_1.png", "case_2.png", "case_3.png"):
                self.assertGreater((root / "cases" / filename).stat().st_size, 100)
            manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen feature provenance"):
                generate_cases(checkpoint, features, manifest, graph_checkpoint, root / "bad")


if __name__ == "__main__":
    unittest.main()
