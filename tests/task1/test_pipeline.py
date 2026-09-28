"""Offline checks for the completed Task 1 pipeline and preserved run."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.task1.data import TAG_VOCAB, build_tag_frame, extract_tags
from src.task1.train import classification_report, load_tag_data, make_split_indices


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "Task1_test" / "files"


class Task1DataTests(unittest.TestCase):
    def test_vocabulary_and_matching(self):
        self.assertEqual(len(TAG_VOCAB), 50)
        tags = extract_tags("Energetic hip-hop with electric guitar, drums and female vocals")
        self.assertTrue({"energetic", "hip hop", "electric guitar", "guitar", "drums"} <= set(tags))
        self.assertIn("female vocals", tags)

    def test_build_tag_frame_is_multihot(self):
        raw = pd.DataFrame(
            {
                "caption": [
                    "calm piano and vocals",
                    "fast rock guitar and drums",
                    "ambient synthesizer",
                ]
            }
        )
        frame, tags, counts = build_tag_frame(raw, top_k=50)
        self.assertEqual(len(frame), 3)
        self.assertEqual(frame.columns.tolist(), ["text", *tags])
        self.assertTrue(set(np.unique(frame[tags].to_numpy())).issubset({0, 1}))
        self.assertEqual(set(tags), set(counts))

    def test_preserved_dataset_schema(self):
        path = ARTIFACTS / "data" / "processed" / "musiccaps_tags.csv"
        texts, labels, tags = load_tag_data(path)
        self.assertEqual(len(texts), 5299)
        self.assertEqual(labels.shape, (5299, 50))
        self.assertEqual(len(tags), 50)
        self.assertGreaterEqual(labels.sum(axis=1).min(), 1)


class Task1SplitAndMetricTests(unittest.TestCase):
    def test_split_is_reproducible_disjoint_and_exhaustive(self):
        first = make_split_indices(100, val_size=0.15, test_size=0.15, seed=42)
        second = make_split_indices(100, val_size=0.15, test_size=0.15, seed=42)
        self.assertEqual(first, second)
        self.assertEqual({name: len(values) for name, values in first.items()}, {
            "train": 70,
            "validation": 15,
            "test": 15,
        })
        flattened = [index for values in first.values() for index in values]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(range(100)))

    def test_classification_report(self):
        targets = np.array([[1, 0], [0, 1], [1, 1]])
        probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6]])
        report = classification_report(targets, probabilities, ["rock", "calm"], 0.5)
        self.assertEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["micro_f1"], 1.0)
        self.assertEqual(report["samples"], 3)
        self.assertEqual(set(report["per_tag"]), {"rock", "calm"})

    def test_preserved_deliverables_and_metadata_agree(self):
        results = ARTIFACTS / "results"
        metrics = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
        metadata = json.loads((results / "model_metadata.json").read_text(encoding="utf-8"))
        examples = json.loads((results / "example_predictions.json").read_text(encoding="utf-8"))
        csv_columns = pd.read_csv(
            ARTIFACTS / "data" / "processed" / "musiccaps_tags.csv", nrows=0
        ).columns.tolist()[1:]
        self.assertEqual(metadata["model_name"], "bert-base-uncased")
        self.assertEqual(metadata["tags"], csv_columns)
        self.assertEqual(len(examples), 5)
        self.assertEqual(len(metrics["history"]["epoch"]), 10)
        self.assertAlmostEqual(metrics["test_macro_f1"], 0.9893633784608408)
        self.assertTrue((results / "f1_curve.png").stat().st_size > 10_000)
        self.assertTrue((results / "best_model.pt").stat().st_size > 400_000_000)


if __name__ == "__main__":
    unittest.main()
