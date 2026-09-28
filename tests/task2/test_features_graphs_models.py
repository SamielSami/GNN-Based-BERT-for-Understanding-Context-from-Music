"""Unit tests for Task 2 features, graph variants, and models."""
from __future__ import annotations

import unittest

import numpy as np
import torch
from torch_geometric.data import Batch

from src.task2.features import (
    AudioFeatureConfig,
    extract_log_mel_from_waveform,
    extract_track_features_from_waveform,
)
from src.task2.graphs import build_edge_variants, build_segment_graph
from src.task2.models import build_task2_model


class Task2FeatureTests(unittest.TestCase):
    def setUp(self):
        self.config = AudioFeatureConfig(
            sample_rate=8_000,
            segment_seconds=0.25,
            n_mels=16,
            n_mfcc=8,
            n_fft=256,
            hop_length=64,
        )
        time = np.arange(8_000, dtype=np.float32) / 8_000
        self.waveform = (0.5 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)

    def test_feature_shapes_and_finiteness(self):
        features = extract_track_features_from_waveform(self.waveform, self.config)
        mel = extract_log_mel_from_waveform(self.waveform, self.config)
        self.assertEqual(features.shape, (4, self.config.node_feature_dim))
        self.assertEqual(mel.shape[0], self.config.n_mels)
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.isfinite(mel).all())
        self.assertGreaterEqual(float(mel.min()), 0.0)
        self.assertLessEqual(float(mel.max()), 1.0)


class Task2GraphAndModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.features = rng.normal(size=(8, 12)).astype(np.float32)

    def test_graph_variants_are_deterministic_and_connected(self):
        first = build_edge_variants(self.features, top_k=2, seed=7)
        second = build_edge_variants(self.features, top_k=2, seed=7)
        self.assertEqual(set(first), {"temporal", "temporal_similarity", "random"})
        for name in first:
            self.assertTrue(torch.equal(first[name], second[name]))
            self.assertFalse(torch.any(first[name][0] == first[name][1]))
        self.assertEqual(first["temporal"].shape[1], 2 * (len(self.features) - 1))
        self.assertEqual(
            first["temporal_similarity"].shape[1], first["random"].shape[1]
        )
        temporal_pairs = set(map(tuple, first["temporal"].t().tolist()))
        for index in range(len(self.features) - 1):
            self.assertIn((index, index + 1), temporal_pairs)
            self.assertIn((index + 1, index), temporal_pairs)

    def test_all_model_output_shapes(self):
        graphs = [
            build_segment_graph(self.features, variant="temporal_similarity", top_k=2),
            build_segment_graph(self.features[:5], variant="temporal", top_k=2),
        ]
        batch = Batch.from_data_list(graphs)
        for name in ("gnn", "mlp"):
            model = build_task2_model(
                name, input_dim=12, num_labels=3, hidden_dim=16, layers=2
            )
            self.assertEqual(model(batch).shape, (2, 3))
        cnn = build_task2_model("cnn", input_dim=12, num_labels=3)
        self.assertEqual(cnn(torch.rand(2, 1, 32, 64)).shape, (2, 3))


if __name__ == "__main__":
    unittest.main()

