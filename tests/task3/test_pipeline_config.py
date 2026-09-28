import tempfile
import unittest
from pathlib import Path

from src.task3.pipeline import ROOT, load_config


class PipelineConfigTests(unittest.TestCase):
    def test_checked_in_config_uses_real_runner_schema_and_paths(self):
        config = load_config("configs/task3.yaml", "results/task3/example_new")
        self.assertEqual(config["output_dir"], ROOT / "results/task3/example_new")
        self.assertEqual(config["modes"], ["bert", "gnn", "concat", "gated", "cross_attention"])
        self.assertNotIn("num_labels", config)
        for name in ("processed_manifest", "source_manifest", "text_checkpoint", "graph_checkpoint"):
            self.assertTrue(config[name].is_absolute())

    def test_rejects_legacy_config_and_unknown_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("training:\n  epochs: 3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown config keys.*missing required keys"):
                load_config(path)
