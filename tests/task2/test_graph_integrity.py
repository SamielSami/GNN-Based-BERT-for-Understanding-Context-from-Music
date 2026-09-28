"""Regressions for invalid audio, segment alignment, graph audits and galleries."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from src.task2.audit import audit_saved_graphs
from src.task2.features import AudioFeatureConfig, extract_log_mel_from_waveform, load_audio, split_segments
from src.task2.graphs import build_edge_variants, validate_graph, validate_graph_variants
from src.task2.preprocess import extract_raw_sample, source_fingerprint
from src.task2.visualize import visualize_graph_examples


class AudioIntervalTests(unittest.TestCase):
    def test_invalid_inputs_are_rejected_and_silence_maps_to_zero(self):
        config = AudioFeatureConfig(sample_rate=4000, n_fft=128, hop_length=32, n_mels=16, n_mfcc=8)
        for waveform in (np.array([]), np.array([np.nan]), np.array([np.inf]), np.zeros((2, 10))):
            with self.subTest(shape=waveform.shape), self.assertRaises(ValueError):
                split_segments(waveform, config)
        for kwargs in ({"segment_seconds": np.nan}, {"segment_seconds": 1e-10}, {"n_mels": 8, "n_mfcc": 20}):
            with self.subTest(config=kwargs), self.assertRaises(ValueError):
                AudioFeatureConfig(**kwargs).validate()
        mel = extract_log_mel_from_waveform(np.zeros(4000, dtype=np.float32), config)
        self.assertTrue(np.array_equal(mel, np.zeros_like(mel)))

    def test_crop_is_shared_by_node_and_cnn_features_and_fingerprinted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "source.wav"
            rate = 4000
            tone = np.sin(2 * np.pi * 300 * np.arange(rate) / rate)
            sf.write(audio, np.concatenate([np.zeros(rate), tone]), rate)
            record = {"sample_id": "tone", "audio_path": str(audio), "split": "train", "labels": [1],
                      "audio_offset_seconds": 1.0, "audio_duration_seconds": 0.5}
            config = AudioFeatureConfig(sample_rate=rate, segment_seconds=0.25, n_fft=128,
                                        hop_length=32, n_mels=16, n_mfcc=8)
            raw = extract_raw_sample(record, root / "manifest.json", config)
            self.assertEqual(raw["audio_samples"], rate // 2)
            self.assertEqual(raw["x_raw"].shape[0], 2)
            self.assertEqual(raw["mel"].shape[-1], 1 + (rate // 2) // 32)
            self.assertGreater(float(raw["mel"].max()), 0)
            old = source_fingerprint(record, root / "manifest.json")
            record["audio_offset_seconds"] = 0
            self.assertNotEqual(old, source_fingerprint(record, root / "manifest.json"))
            with self.assertRaisesRegex(ValueError, "too short"):
                load_audio(audio, rate, offset_seconds=1.5, duration_seconds=1)


class GraphIntegrityTests(unittest.TestCase):
    def test_graph_validator_rejects_disconnection_and_malformed_edges(self):
        x = torch.ones(4, 2)
        invalid = (
            (torch.tensor([[0, 1], [1, 0]]), "disconnected"),
            (torch.tensor([[0, 4], [4, 0]]), "out-of-range"),
            (torch.tensor([[0, 1], [1, 2]]), "symmetric"),
            (torch.tensor([[0, 0], [0, 0]]), "duplicate"),
            (torch.tensor([[0.0, 1.0], [1.0, 0.0]]), "integer"),
        )
        for edges, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_graph(x, edges)
        self.assertTrue(validate_graph(torch.ones(1, 2), torch.empty(2, 0, dtype=torch.long))["connected"])
        variants = build_edge_variants(np.arange(24, dtype=np.float32).reshape(8, 3), top_k=1)
        variants["random"] = variants["temporal"]
        with self.assertRaisesRegex(ValueError, "identical edge counts"):
            validate_graph_variants(torch.ones(8, 3), variants)

    def test_saved_audit_and_twenty_distinct_three_panel_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x = torch.from_numpy(np.random.default_rng(42).normal(size=(6, 5)).astype(np.float32))
            records = []
            for index in range(21):
                split = ("train", "validation", "test")[index % 3]
                sample_id = f"synthetic_graph_{index:02d}"
                sample_path = root / f"{sample_id}.pt"
                torch.save({"sample_id": sample_id, "split": split, "labels": torch.tensor([1, 0]),
                            "x": x, "mel": torch.zeros(1, 8, 16),
                            "edge_indices": build_edge_variants(x.numpy(), top_k=1, seed=index)}, sample_path)
                records.append({"sample_id": sample_id, "split": split, "processed_path": sample_path.name})
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"label_names": ["a", "b"], "records": records}), encoding="utf-8")
            audit = audit_saved_graphs(manifest_path)
            self.assertTrue(audit["valid"])
            self.assertEqual(audit["graphs_checked"], 63)
            self.assertTrue(audit["all_splits_present"])
            gallery = visualize_graph_examples(manifest_path, root / "gallery", count=20)
            self.assertTrue(gallery["at_least_20_examples"])
            self.assertEqual(gallery["graph_panels_written"], 60)
            self.assertEqual(len({item["sample_id"] for item in gallery["examples"]}), 20)
            self.assertEqual(len(list((root / "gallery").glob("*.png"))), 20)
            self.assertTrue((root / "gallery" / "index.html").is_file())
            damaged_path = root / records[0]["processed_path"]
            damaged = torch.load(damaged_path, weights_only=False)
            damaged["x"][0, 0] = float("nan")
            torch.save(damaged, damaged_path)
            audit = audit_saved_graphs(manifest_path)
            self.assertFalse(audit["valid"])
            self.assertEqual(audit["errors"][0]["sample_id"], records[0]["sample_id"])


if __name__ == "__main__":
    unittest.main()
