"""Structural checks for the Task 1 Jupyter notebook."""
from __future__ import annotations

import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "task1_bert_baseline.ipynb"


class Task1NotebookTests(unittest.TestCase):
    def test_notebook_has_expected_workflow_and_safe_defaults(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("RUN_INFERENCE = True", code)
        self.assertIn("RUN_TRAINING = False", code)
        self.assertIn("TRAIN_EPOCHS = 10 if TRAIN_DEVICE == 'cuda' else 2", code)
        self.assertIn("Task1Config", code)
        self.assertIn("def prediction_pairs(items):", code)
        self.assertIn("isinstance(item, dict)", code)
        self.assertIn("prediction_pairs(example['top_predicted'])", code)
        self.assertIn("WARNING: active checkpoint and metrics are from different runs", code)
        self.assertIn("load_prepared_manifest(DATA_PATH)", code)
        self.assertIn("splits = manifest['split_indices']", code)
        self.assertIn("metadata['thresholds'] == metrics['calibration']['thresholds']", code)
        self.assertIn("audioset_cpu_20260907", code)
        self.assertNotIn("splits = make_split_indices(", code)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell_{index}", "exec")

    def test_example_cell_handles_legacy_and_current_predictions(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = next("".join(cell["source"]) for cell in notebook["cells"]
                    if "enumerate(saved_examples" in "".join(cell["source"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.json"
            for predictions in ([["rock", 0.9]], [{"tag": "rock", "probability": 0.9}]):
                path.write_text(json.dumps([{"text": "rock music", "true_tags": ["rock"],
                                             "top_predicted": predictions}]), encoding="utf-8")
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(code, {"EXAMPLES_PATH": path})
                self.assertIn("('rock', 0.9)", output.getvalue())

    def test_example_cell_runs_without_setup(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = next("".join(cell["source"]) for cell in notebook["cells"]
                    if "enumerate(saved_examples" in "".join(cell["source"]))
        from unittest.mock import patch
        for working_directory in (ROOT, ROOT / "notebooks"):
            scope = {}
            with patch("pathlib.Path.cwd", return_value=working_directory), contextlib.redirect_stdout(io.StringIO()):
                exec(code, scope)
            self.assertEqual(len(scope["saved_examples"]), 5)
            self.assertEqual(scope["prediction_pairs"](["rock"]), ["rock"])


if __name__ == "__main__":
    unittest.main()
