"""Offline regressions for training configuration, frozen BERT, and metrics."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.task1.train import BertTagClassifier, Task1Config, classification_report


class TrainingSafetyTests(unittest.TestCase):
    def test_invalid_settings_fail_before_training(self):
        for kwargs in ({"epochs": 0}, {"batch_size": 0}, {"patience": 0},
                       {"threshold": 2}, {"val_size": 0.9, "test_size": 0.2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Task1Config(data_path="unused.csv", **kwargs)

    def test_frozen_encoder_stays_in_eval_but_head_trains(self):
        encoder = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(4, 4))
        encoder.config = SimpleNamespace(hidden_size=4)
        with patch("src.task1.train.AutoModel.from_pretrained", return_value=encoder):
            model = BertTagClassifier("offline", 2, freeze_bert=True)
        model.eval()
        model.train()
        self.assertFalse(model.bert.training)
        self.assertTrue(model.dropout.training)
        self.assertTrue(model.classifier.training)
        self.assertTrue(all(not p.requires_grad for p in model.bert.parameters()))

    def test_average_precision_includes_all_positive_labels(self):
        report = classification_report(
            np.array([[1, 0], [1, 0]]), np.array([[0.8, 0.1], [0.7, 0.2]]),
            ["present", "absent"], 0.5,
        )
        self.assertEqual(report["per_tag"]["present"]["auc_pr"], 1.0)
        self.assertIsNone(report["per_tag"]["absent"]["auc_pr"])
        self.assertEqual(report["mean_auc_pr"], 1.0)
