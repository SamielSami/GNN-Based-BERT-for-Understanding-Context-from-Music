"""Integration checks for frozen validation selection and auditable suite evaluation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.task3.data import make_demo
from src.task3.experiment import evaluate_experiment, run_experiment, verify_experiment
from src.task3.models import MODES


class ExperimentTests(unittest.TestCase):
    def test_three_seed_suite_freezes_selection_and_verifies_saved_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = make_demo(root / "features.pt")
            output = root / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                report = run_experiment(features, output, epochs=1, hidden_dim=8, device="cpu", cpu_threads=1)
            self.assertFalse(report["evaluated_test"])
            self.assertEqual(len(report["rows"]), 15)
            self.assertEqual(list(report["selection"]["validation_means"]), list(MODES))
            selection_before = (output / "selection.json").read_bytes()
            self.assertFalse(list(output.glob("seed*/*_test_predictions.npz")))
            aggregate = pd.read_csv(output / "comparison_aggregate.csv").set_index("mode")
            for mode in MODES:
                values = [row["validation_macro_f1"] for row in report["rows"] if row["mode"] == mode]
                self.assertAlmostEqual(aggregate.loc[mode, "validation_macro_f1_std"], np.std(values, ddof=1))
            with patch("src.task3.experiment.run_experiments", side_effect=AssertionError("Must not retrain")), \
                    contextlib.redirect_stdout(io.StringIO()):
                report = evaluate_experiment(output, device="cpu", cpu_threads=1)
            self.assertTrue(report["evaluated_test"])
            self.assertEqual((output / "selection.json").read_bytes(), selection_before)
            self.assertEqual(len(list(output.glob("seed*/*_test_predictions.npz"))), 15)
            verified = verify_experiment(output)
            self.assertTrue(verified["valid"])
            self.assertTrue(verified["suite_declaration_verified"])
            self.assertTrue(verified["comparison_tables_verified"])
            self.assertEqual(verified["runs_checked"], 15)
            self.assertTrue(all(check["test_recomputed"] for check in verified["checks"]))
            with self.assertRaises(FileExistsError):
                evaluate_experiment(output, device="cpu")
            with self.assertRaises(FileExistsError):
                run_experiment(features, output)
            selection = json.loads(selection_before)
            selection["selected_mode"] = "corrupted"
            (output / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selection changed"):
                verify_experiment(output)
            (output / "selection.json").write_bytes(selection_before)
            test_path = output / report["rows"][0]["test_predictions"]
            with np.load(test_path) as saved:
                changed = {name: saved[name].copy() for name in saved.files}
            changed["targets"][0, 0] = 1 - changed["targets"][0, 0]
            np.savez_compressed(test_path, **changed)
            with self.assertRaisesRegex(ValueError, "Test predictions changed"):
                verify_experiment(output)

    def test_saved_tables_and_declaration_cannot_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = make_demo(root / "features.pt")
            output = root / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                run_experiment(features, output, seeds=[42], epochs=1, hidden_dim=8,
                               device="cpu", cpu_threads=1)
            aggregate_path = output / "comparison_aggregate.csv"
            aggregate = pd.read_csv(aggregate_path)
            self.assertTrue(aggregate["validation_macro_f1_std"].isna().all())
            self.assertTrue(verify_experiment(output)["valid"])

            # Reported means, sample std, and individual measurements all belong
            # to the audit, including the undefined std of a one-seed fixture.
            for filename, column, value in (
                ("comparison.csv", "validation_macro_f1", 999.0),
                ("comparison_aggregate.csv", "validation_macro_f1_mean", 999.0),
                ("comparison_aggregate.csv", "validation_macro_f1_std", 0.0),
            ):
                with self.subTest(filename=filename, column=column):
                    path = output / filename
                    original = path.read_bytes()
                    frame = pd.read_csv(path)
                    frame.loc[0, column] = value
                    frame.to_csv(path, index=False)
                    with self.assertRaisesRegex(ValueError, f"Saved {filename}"):
                        verify_experiment(output)
                    with patch("src.task3.experiment.predict_features", side_effect=AssertionError("No test inference")):
                        with self.assertRaisesRegex(ValueError, f"Saved {filename}"):
                            evaluate_experiment(output, device="cpu", cpu_threads=1)
                    path.write_bytes(original)

            for filename, key, value, message in (
                ("suite_config.json", "seeds", [999], "suite configuration changed"),
                ("experiment.json", "selection", {}, "Embedded validation selection"),
            ):
                with self.subTest(filename=filename):
                    path = output / filename
                    original = path.read_bytes()
                    content = json.loads(original)
                    content[key] = value
                    path.write_text(json.dumps(content), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        verify_experiment(output)
                    path.write_bytes(original)
            self.assertTrue(verify_experiment(output)["valid"])

    def test_bad_checkpoint_rejected_before_test_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = make_demo(root / "features.pt")
            with contextlib.redirect_stdout(io.StringIO()):
                run_experiment(features, root / "run", seeds=[42], epochs=1, hidden_dim=8,
                               device="cpu", cpu_threads=1)
            with (root / "run/seed42/concat.pt").open("ab") as stream:
                stream.write(b"changed")
            with patch("src.task3.experiment.predict_features", side_effect=AssertionError("No test inference")):
                with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                    evaluate_experiment(root / "run", device="cpu", cpu_threads=1)


if __name__ == "__main__":
    unittest.main()
