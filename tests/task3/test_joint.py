"""Offline joint-training checks using tiny actual BERT and GraphSAGE models."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import AutoModel, BertConfig
from torch_geometric.data import Data, Batch

from src.task2.models import GraphSAGEEncoder
from src.task3 import joint
from src.task3.models import MODES
from tests.task3.test_prepare import make_fixture, TinyTokenizer


class SavingTokenizer(TinyTokenizer):
    def save_pretrained(self, path):
        Path(path).mkdir()
        (Path(path) / "tokenizer_config.json").write_text("{}")


class JointTests(unittest.TestCase):
    def test_partial_encoder_gradients_masks_and_frozen_parameters(self):
        torch.set_num_threads(1)
        config = BertConfig(vocab_size=32, hidden_size=8, num_hidden_layers=2,
                            num_attention_heads=2, intermediate_size=16, max_position_embeddings=8)
        dims = dict(text_dim=8, graph_dim=8, num_labels=2, hidden_dim=8)
        graphs = Batch.from_data_list([Data(x=torch.randn(3, 3), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])) for _ in range(2)])
        tokens = dict(input_ids=torch.tensor([[1, 4, 2, 0], [1, 5, 2, 0]]),
                      attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]))
        for mode in MODES:
            with self.subTest(mode=mode):
                model = joint.JointClassifier(AutoModel.from_config(config), GraphSAGEEncoder(3, 4, 1, 0), dims, mode, 1)
                before = joint.frozen_text_hash(model)
                anchors = joint.parameter_anchors(model)
                model.train()
                if model.text_encoder:
                    self.assertFalse(model.text_encoder.embeddings.training)
                    self.assertFalse(model.text_encoder.encoder.layer[0].training)
                    self.assertTrue(model.text_encoder.encoder.layer[1].training)
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
                loss = model(graphs, tokens).square().mean()
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    else:
                        self.assertIsNone(parameter.grad)
                opt.step()
                self.assertEqual(before, joint.frozen_text_hash(model))
                self.assertTrue(all(v['max_abs_change'] > 0 for v in joint.changed_parameters(model, anchors).values()))
                model.eval()
                changed = {k: v.clone() for k, v in tokens.items()}
                changed['input_ids'][:, -1] = 19
                torch.testing.assert_close(model(graphs, tokens), model(graphs, changed), atol=1e-6, rtol=1e-5)
                single = {k: v[:1] for k, v in tokens.items()}
                torch.testing.assert_close(model(graphs, tokens)[:1], model(graphs.to_data_list()[0], single), atol=1e-6, rtol=1e-5)

    def test_full_joint_train_restore_evaluate_and_tamper_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = make_fixture(root)
            output = root / "joint"
            with patch.object(joint.AutoConfig, "from_pretrained", return_value=fixture.config), \
                 patch.object(joint.AutoTokenizer, "from_pretrained", return_value=SavingTokenizer()), \
                 contextlib.redirect_stdout(io.StringIO()):
                initial_hash = joint.file_hash(fixture.text_checkpoint)
                result = joint.train_joint(fixture.processed_path, fixture.source_path, fixture.text_checkpoint,
                                           fixture.graph_checkpoint, output, epochs=2, batch_size=2,
                                           hidden_dim=8, device="cpu", cpu_threads=1)
                self.assertEqual(len(result['runs']), 5)
                self.assertEqual(initial_hash, joint.file_hash(fixture.text_checkpoint))
                self.assertFalse(list(output.glob('*/test_predictions.npz')))
                self.assertTrue(joint.verify_joint(output)['valid'])
                selected_before = (output / 'selection.json').read_bytes()
                for mode in MODES:
                    saved = torch.load(output / mode / 'best_model.pt', weights_only=True)
                    restored = joint.restore_model(saved)
                    data = joint.PairedDataset(fixture.processed_path, fixture.source, 'validation',
                                               'temporal_similarity', SavingTokenizer(), 4)
                    actual = joint.predict(restored, data, 2, 'cpu')
                    with np.load(output / mode / 'validation_predictions.npz') as archive:
                        np.testing.assert_allclose(actual['probabilities'], archive['probabilities'], atol=1e-6)
                verified = joint.evaluate_joint(output, device='cpu', cpu_threads=1)
                self.assertTrue(verified['evaluated_test'])
                self.assertEqual(selected_before, (output / 'selection.json').read_bytes())
                with self.assertRaises(FileExistsError):
                    joint.evaluate_joint(output, device='cpu', cpu_threads=1)
                with (output / 'concat/best_model.pt').open('ab') as stream:
                    stream.write(b'changed')
                with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                    joint.verify_joint(output)

    def test_alignment_and_positive_support_pr_auc(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_fixture(Path(directory))
            source = json.loads(fixture.source_path.read_text())
            source['records'][0]['text'] = ''
            fixture.source_path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'caption'):
                joint.aligned_manifests(fixture.processed_path, fixture.source_path)
        metrics = joint.metric_report(np.array([[1, 0], [0, 0]]), np.array([[.9, .1], [.1, .2]]), ['a', 'b'], .5)
        self.assertEqual(metrics['mean_pr_auc_trapezoidal'], 1.0)
        self.assertIsNone(metrics['per_label']['b']['pr_auc_trapezoidal'])


if __name__ == '__main__':
    unittest.main()
