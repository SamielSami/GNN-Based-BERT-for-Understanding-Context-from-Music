import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.task3.semantic_plots import label_categories, plot_semantic_embeddings


class SemanticPlotTests(unittest.TestCase):
    def test_overlap_unknown_and_true_annotation_mapping(self):
        names = ["/m/06by7", "/m/064t9", "/t/dd00036"]
        mapping = {"genre": {names[0]: "Rock music", names[1]: "Pop music"}, "mood": {names[2]: "Angry music"}}
        targets = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 1]])
        active, categories, _ = label_categories(targets, names, mapping, 'genre')
        self.assertEqual(active[0], ['Rock music', 'Pop music'])
        self.assertEqual(categories, ['Multiple selected genre labels', 'No selected genre annotation', 'Rock music'])
        _, mood, _ = label_categories(targets, names, mapping, 'mood')
        self.assertEqual(mood, ['No selected mood annotation', 'No selected mood annotation', 'Angry music'])

    def test_plot_artifacts_preserve_ids_coordinates_and_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'predictions.npz'
            ids = np.array(['a', 'b', 'c', 'd'])
            targets = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 1, 0]])
            np.savez(path, sample_ids=ids, targets=targets, embeddings=np.random.default_rng(42).normal(size=(4, 8)))
            report = plot_semantic_embeddings(path, ['/m/06by7', '/m/064t9', '/t/dd00036'], root / 'plots', 'Synthetic fixture')
            self.assertEqual(report['genre']['overlapping_samples'], 1)
            self.assertEqual(report['mood']['annotated_samples'], 1)
            self.assertEqual(report['mood']['unannotated_samples'], 3)
            for name in ['genre_mood_tsne.png', 'genre_label_panels.png', 'sample_annotations.csv', 'metadata.json', 'README.md']:
                self.assertTrue((root / 'plots' / name).is_file())
            with np.load(root / 'plots/embedding_coordinates.npz') as archive:
                np.testing.assert_array_equal(archive['sample_ids'], ids)
                np.testing.assert_array_equal(archive['targets'], targets)
                self.assertEqual(archive['coordinates'].shape, (4, 2))
            with self.assertRaises(FileExistsError):
                plot_semantic_embeddings(path, ['/m/06by7', '/m/064t9', '/t/dd00036'], root / 'plots', 'Fixture')
