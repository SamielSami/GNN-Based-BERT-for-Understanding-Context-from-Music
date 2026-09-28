"""Offline unit tests for the Part 1 MusicCaps audit."""
from __future__ import annotations

import unittest

import pandas as pd

from src.task1.audit import (
    build_metadata_audit,
    canonicalize_label,
    parse_list,
    phrase_in_text,
    summarize_availability,
)


class MusicCapsAuditTests(unittest.TestCase):
    def test_list_parsing_and_normalization(self):
        self.assertEqual(parse_list("['soft piano', 'female vocal']"), ["soft piano", "female vocal"])
        self.assertEqual(parse_list("/m/01,/m/02"), ["/m/01", "/m/02"])
        self.assertEqual(canonicalize_label("R&B / Soul"), "r and b soul")
        self.assertTrue(phrase_in_text("soft piano", "A SOFT piano melody plays."))
        self.assertFalse(phrase_in_text("rock", "A rocket sound."))

    def test_metadata_audit_detects_duplicates_and_overlap(self):
        frame = pd.DataFrame(
            {
                "ytid": ["a", "a", "b"],
                "start_s": [0, 10, 0],
                "end_s": [10, 20, 10],
                "audioset_positive_labels": ["/m/1,/m/2", "/m/1", "/m/2"],
                "aspect_list": ["['soft piano']", "['rock']", "['drums']"],
                "caption": ["Soft piano music", "A guitar clip", "Drums play"],
                "author_id": [1, 1, 2],
                "is_balanced_subset": [True, False, False],
                "is_audioset_eval": [True, True, False],
            }
        )
        audit = build_metadata_audit(frame, top_k=2)
        self.assertEqual(audit["metadata"]["rows"], 3)
        self.assertEqual(audit["metadata"]["duplicate_video_id_rows"], 2)
        self.assertEqual(audit["metadata"]["duplicate_clip_window_rows"], 0)
        self.assertEqual(audit["lexical_overlap"]["rows_with_any_exact_aspect_phrase_in_caption"], 2)
        self.assertEqual(
            audit["label_analysis"]["top_audioset_label_ids_global_audit_only"][0]["label"],
            "/m/1",
        )

    def test_availability_gate_handles_technical_errors(self):
        inconclusive = summarize_availability(
            [
                {"status": "available"},
                {"status": "technical_error"},
            ],
            threshold=0.8,
        )
        self.assertEqual(inconclusive["gate"], "inconclusive")
        passing = summarize_availability(
            [{"status": "available"}] * 9 + [{"status": "unavailable"}],
            threshold=0.8,
        )
        self.assertEqual(passing["gate"], "pass")


if __name__ == "__main__":
    unittest.main()
