"""Regressions for independent labels, inherited splits, and explicit audio crops."""
from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.task2.manifest import build_aligned_musiccaps_manifest, main


class ManifestAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tags = ["/m/second", "/m/first"]  # Preserve source order, not lexical order.
        self.ids = ["train_a", "train_b", "validation_a", "test_a"]
        self.metadata = pd.DataFrame({
            "ytid": self.ids, "start_s": [30.5, 40, 10, 0], "end_s": [40.5, 50, 20, 10],
            "caption": ["duplicate caption"] * 4,
            "audioset_positive_labels": ["/m/first", "/m/second", "/m/first,/m/second", ""],
        })
        self.labels = pd.DataFrame({
            "ytid": self.ids, "text": ["original text"] * 4,
            self.tags[0]: [0, 1, 1, 0], self.tags[1]: [1, 0, 1, 0],
        })
        self.source = {
            "label_source": "MusicCaps.audioset_positive_labels", "vocabulary_fit_split": "train",
            "split_before_vocabulary": True, "tags": self.tags,
            "split_ids": {"train": self.ids[:2], "validation": [self.ids[2]], "test": [self.ids[3]]},
            "train_support": {self.tags[0]: 1, self.tags[1]: 1}, "seed": 42, "rows": 4,
        }

    def build(self, directory: str, **kwargs):
        return build_aligned_musiccaps_manifest(
            kwargs.pop("metadata", self.metadata), kwargs.pop("labels", self.labels),
            task1_manifest=kwargs.pop("source", self.source), audio_dir=Path(directory), **kwargs,
        )

    def test_available_audio_intersection_inherits_ids_splits_and_zero_positive_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            for video_id in ["train_a", "validation_a", "test_a"]:
                (Path(directory) / f"{video_id}.wav").touch()
            manifest = self.build(directory)
        self.assertEqual(manifest["label_names"], self.tags)
        self.assertEqual(manifest["audit"]["split_counts"], {"train": 1, "validation": 1, "test": 1})
        self.assertTrue(manifest["audit"]["ready_for_training"])
        self.assertEqual(manifest["audit"]["missing_sample_ids"], ["train_b_40_50"])
        by_id = {record["ytid"]: record for record in manifest["records"]}
        self.assertEqual(by_id["train_a"]["labels"], [0, 1])
        self.assertEqual(by_id["test_a"]["labels"], [0, 0])
        self.assertEqual(by_id["validation_a"]["split"], "validation")
        self.assertEqual(by_id["train_a"]["text"], "duplicate caption")

    def test_input_row_order_cannot_change_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.build(directory, include_missing=True)
            shuffled = self.build(
                directory, include_missing=True,
                metadata=self.metadata.iloc[::-1], labels=self.labels.iloc[[2, 0, 3, 1]],
            )
        self.assertEqual(first["records"], shuffled["records"])
        self.assertFalse(first["audit"]["ready_for_training"])

    def test_audio_mode_preserves_fractional_crop_and_prefers_explicit_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "train_a.wav"
            source_path.touch()
            clip_path = Path(directory) / "train_a_30.5_40.5.wav"
            clip_path.touch()
            trimmed = self.build(directory)["records"][0]
            source = self.build(directory, audio_mode="full_source")["records"][0]
        self.assertEqual(trimmed["audio_path"], str(clip_path.resolve()))
        self.assertEqual(source["audio_path"], str(source_path.resolve()))
        self.assertEqual(trimmed["audio_offset_seconds"], 0.0)
        self.assertEqual(source["audio_offset_seconds"], 30.5)
        self.assertEqual(source["audio_duration_seconds"], 10.0)
        self.assertEqual(trimmed["sample_id"], "train_a_30.5_40.5")

    def test_full_source_does_not_apply_offset_to_segment_named_file(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "train_a_30.5_40.5.wav").touch()
            manifest = self.build(directory, audio_mode="full_source")
        self.assertEqual(manifest["records"], [])

    def test_split_overlap_and_missing_source_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            overlap = copy.deepcopy(self.source)
            overlap["split_ids"]["test"].append("train_a")
            with self.assertRaisesRegex(ValueError, "overlap"):
                self.build(directory, source=overlap)
            with self.assertRaisesRegex(ValueError, "exactly match"):
                self.build(directory, labels=self.labels.iloc[:-1])

    def test_nonbinary_labels_and_source_order_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = self.labels.copy()
            labels.loc[0, self.tags[0]] = float("nan")
            with self.assertRaisesRegex(ValueError, "finite binary"):
                self.build(directory, labels=labels)
            with self.assertRaisesRegex(ValueError, "schema/tag order"):
                self.build(directory, labels=self.labels[["ytid", "text", *self.tags[::-1]]])

    def test_source_provenance_and_official_labels_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            source = dict(self.source, label_source="caption_keyword_pseudo_labels")
            with self.assertRaisesRegex(ValueError, "independent AudioSet"):
                self.build(directory, source=source)
            metadata = self.metadata.copy()
            metadata.loc[0, "audioset_positive_labels"] = "/m/second"
            with self.assertRaisesRegex(ValueError, "disagree with official"):
                self.build(directory, metadata=metadata)

    def test_duplicate_ids_and_invalid_intervals_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = pd.concat([self.metadata, self.metadata.iloc[[0]]], ignore_index=True)
            with self.assertRaisesRegex(ValueError, "one MusicCaps interval"):
                self.build(directory, metadata=metadata)
            for start, end in [(10, 10), (-1, 10), (0, float("inf")), (float("nan"), 10)]:
                with self.subTest(start=start, end=end):
                    metadata = self.metadata.copy()
                    metadata.loc[0, ["start_s", "end_s"]] = [start, end]
                    with self.assertRaisesRegex(ValueError, "Invalid MusicCaps interval"):
                        self.build(directory, metadata=metadata)

    def test_cli_verifies_checksum_and_requires_explicit_audio_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, labels, source = root / "metadata.csv", root / "labels.csv", root / "source.json"
            self.metadata.to_csv(metadata, index=False)
            self.labels.to_csv(labels, index=False)
            provenance = dict(self.source, csv_sha256=hashlib.sha256(labels.read_bytes()).hexdigest())
            source.write_text(json.dumps(provenance), encoding="utf-8")
            arguments = ["manifest", "--metadata-csv", str(metadata), "--labels-csv", str(labels),
                         "--task1-manifest", str(source), "--audio-dir", directory,
                         "--output", str(root / "output.json"), "--missing-log", str(root / "missing.json")]
            with patch("sys.argv", arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
            arguments.extend(["--audio-mode", "pretrimmed"])
            # An empty official-label string is a legitimate zero-positive example.
            # read_csv represents it as NaN, so preserve it explicitly in this fixture.
            self.metadata.loc[3, "audioset_positive_labels"] = "[]"
            self.metadata.to_csv(metadata, index=False)
            with patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                main()
            result = json.loads((root / "output.json").read_text(encoding="utf-8"))
            self.assertTrue(result["task1_alignment"]["source_csv_hash_verified"])
            labels.write_text(labels.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with patch("sys.argv", arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()


if __name__ == "__main__":
    unittest.main()
