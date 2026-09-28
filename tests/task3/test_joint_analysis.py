"""Checks for fixed-case selection and saved-prediction failure analysis."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.task3.joint import metric_report, write_json
from src.task3.joint_analyze import analyze_joint, sample_f1
from src.task3.models import MODES


class JointAnalysisTests(unittest.TestCase):
    def test_empty_label_example_has_finite_zero_f1(self):
        values = sample_f1(np.array([[0, 0], [1, 0]], dtype=bool), np.array([[0, 0], [1, 1]], dtype=bool))
        np.testing.assert_allclose(values, [0, 2 / 3])

    def test_cases_keep_manifest_order_and_controls_use_saved_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = np.array(['z', 'a', 'm', 'b'])
            labels = ['label1', 'label2']
            targets = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
            source = root / 'source.json'
            write_json(source, dict(records=[dict(sample_id=s, ytid=s, text=f'caption {s}', labels=y.tolist())
                                              for s, y in zip(ids.tolist(), targets)]))
            write_json(root / 'run_config.json', dict(source_manifest=str(source), modes=list(MODES),
                       label_names=labels, seed=42, epochs=3, patience=2, batch_size=8, hidden_dim=64,
                       head_initialization='Test fixture', unfreeze_layers=1))
            write_json(root / 'selection.json', dict(selected_fusion='concat', selected_mode='bert'))
            rows = []
            for mode in MODES:
                directory = root / mode
                directory.mkdir()
                threshold = .7 if mode == 'bert' else .4
                probabilities = np.array([[.6, .1], [.1, .6], [.8, .5], [.1, .2]])
                row = dict(mode=mode, best_epoch=1, threshold=threshold)
                for split in ('validation', 'test'):
                    np.savez_compressed(directory / f'{split}_predictions.npz', sample_ids=ids,
                                        targets=targets, probabilities=probabilities)
                    metrics = metric_report(targets, probabilities, labels, threshold)
                    write_json(directory / f'{split}_metrics.json', metrics)
                    row.update({f'{split}_{k}': metrics[k] for k in ['macro_f1', 'micro_f1', 'mean_average_precision', 'mean_pr_auc_trapezoidal']})
                rows.append(row)
            pd.DataFrame(rows).to_csv(root / 'comparison.csv', index=False)
            with patch('src.task3.joint_analyze.verify_joint', return_value=dict(evaluated_test=True)):
                evidence = analyze_joint(root)
                self.assertEqual(evidence['case_ids'], ['z', 'a', 'm'])
                cases = json.loads((root / 'analysis/cases.json').read_text())
                self.assertEqual(cases[0]['models']['bert']['threshold'], .7)
                self.assertEqual(cases[0]['models']['concat']['labels'][0]['status'], 'true_positive')
                self.assertEqual(cases[0]['models']['bert']['labels'][0]['status'], 'false_negative')
                failure = json.loads((root / 'analysis/failure_summary.json').read_text())
                self.assertEqual(failure['controls']['bert']['fusion_better'], 3)
                self.assertTrue((root / 'analysis/ablation.png').is_file())
                with self.assertRaises(FileExistsError):
                    analyze_joint(root)


if __name__ == '__main__':
    unittest.main()
