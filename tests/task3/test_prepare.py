"""Regression checks for encoder provenance and memory-mapped feature extraction."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, BertConfig

from src.task1.audioset import LABEL_SOURCE
from src.task2.dataset import dataset_identity, load_manifest
from src.task2.models import build_task2_model
from src.task3 import prepare
from src.task3.data import load_features


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def make_fixture(root):
    """Build real tiny encoder checkpoints and caches without downloading models."""
    text_dir, graph_dir = root / "text", root / "processed"
    text_dir.mkdir()
    graph_dir.mkdir()
    labels = ["/m/first", "/m/second"]
    splits = {"train": [0, 1], "validation": [2], "test": [3]}
    frame = pd.DataFrame({
        "ytid": [f"video{i}" for i in range(4)],
        "text": [f"caption {i}" for i in range(4)],
        labels[0]: [1, 0, 1, 0], labels[1]: [0, 1, 1, 0],
    })
    csv_path = text_dir / "data.csv"
    frame.to_csv(csv_path, index=False)
    prepared = {
        "label_source": LABEL_SOURCE, "vocabulary_fit_split": "train",
        "split_before_vocabulary": True, "tags": labels,
        "csv_sha256": prepare.file_hash(csv_path), "split_indices": splits,
        "split_ids": {split: [f"video{i}" for i in indices] for split, indices in splits.items()},
        "train_support": dict.fromkeys(labels, 1),
    }
    write_json(csv_path.with_suffix(".manifest.json"), prepared)
    write_json(text_dir / "dataset_manifest.json", prepared)
    write_json(text_dir / "split_indices.json", splits)
    source_records = [
        dict(sample_id=f"sample{i}", ytid=f"video{i}", text=f"caption {i}",
             labels=[int(value) for value in frame.iloc[i][labels]], split=split)
        for split, i in (("validation", 2), ("train", 0), ("test", 3))
    ]
    source = dict(label_names=labels, label_source=LABEL_SOURCE, records=source_records)
    source_path = root / "source.json"
    write_json(source_path, source)
    processed_records = []
    for row in reversed(source_records):
        filename = f"{row['sample_id']}.pt"
        torch.save({
            "sample_id": row["sample_id"], "split": row["split"], "labels": row["labels"],
            "x": torch.tensor([[1., 2., 3.], [3., 2., 1.]]), "mel": torch.ones(2, 3),
            "edge_indices": {"temporal_similarity": torch.tensor([[0, 1], [1, 0]])},
        }, graph_dir / filename)
        processed_records.append(dict(row, processed_path=filename))
    torch.save({"mean": torch.zeros(3), "std": torch.ones(3)}, graph_dir / "normalization.pt")
    processed = dict(label_names=labels, label_source=LABEL_SOURCE, records=processed_records,
                     feature_config={"sample_rate": 16000}, normalization_path="normalization.pt")
    processed_path = graph_dir / "manifest.json"
    write_json(processed_path, processed)
    config = BertConfig(vocab_size=32, hidden_size=8, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=16, max_position_embeddings=8)
    text_model = AutoModel.from_config(config).eval()
    text_saved = dict(config={"data_path": str(csv_path), "text_col": "text",
                              "model_name": "offline-tiny-bert", "max_length": 4},
                      dataset_sha256=prepare.file_hash(csv_path), label_source=LABEL_SOURCE,
                      tags=labels, state_dict={f"bert.{key}": value for key, value in text_model.state_dict().items()})
    text_checkpoint = text_dir / "best.pt"
    torch.save(text_saved, text_checkpoint)
    graph_model = build_task2_model("gnn", input_dim=3, num_labels=2, hidden_dim=4, layers=1, dropout=0)
    graph_saved = dict(config={"model": "gnn", "manifest": str(processed_path),
                               "graph_variant": "temporal_similarity", "hidden_dim": 4,
                               "layers": 1, "dropout": 0}, input_dim=3, label_names=labels,
                       dataset_fingerprint=dataset_identity(processed_path)["sha256"],
                       split_ids={split: [row["sample_id"] for row in processed_records if row["split"] == split]
                                  for split in prepare.SPLITS}, state_dict=graph_model.state_dict())
    graph_checkpoint = root / "graph.pt"
    torch.save(graph_saved, graph_checkpoint)
    return SimpleNamespace(**locals())


class TinyTokenizer:
    def __call__(self, texts, *, truncation, padding, max_length, return_tensors):
        assert truncation and padding == "max_length" and max_length == 4 and return_tensors == "pt"
        ids = torch.tensor([[1, 4 + int(text.rsplit(" ", 1)[1]), 2, 0] for text in texts])
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}

    def convert_ids_to_tokens(self, ids):
        return [f"token_{value}" for value in ids]


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.fixture = make_fixture(self.root)

    def test_meta_encoder_restores_nonpersistent_buffers_and_exact_weights(self):
        fixture = self.fixture
        saved = torch.load(fixture.text_checkpoint, weights_only=True, mmap=True)
        with patch.object(prepare.AutoConfig, "from_pretrained", return_value=fixture.config):
            restored = prepare._load_text_encoder(saved, torch.device("cpu"))
        self.assertFalse(restored.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in restored.parameters()))
        self.assertTrue(all(not buffer.is_meta for buffer in restored.buffers()))
        torch.testing.assert_close(restored.embeddings.position_ids, torch.arange(8).unsqueeze(0))
        torch.testing.assert_close(restored.embeddings.token_type_ids, torch.zeros(1, 8, dtype=torch.long))
        inputs = TinyTokenizer()(["caption 0"], truncation=True, padding="max_length", max_length=4, return_tensors="pt")
        with torch.inference_mode():
            torch.testing.assert_close(restored(**inputs).last_hidden_state,
                                       fixture.text_model(**inputs).last_hidden_state)

    def test_graph_relocation_requires_exact_cached_dataset(self):
        fixture = self.fixture
        relocated = self.root / "relocated"
        shutil.copytree(fixture.graph_dir, relocated)
        path = relocated / "manifest.json"
        saved = copy.deepcopy(fixture.graph_saved)
        saved["config"]["manifest"] = str(self.root / "unavailable-training-path.json")
        self.assertEqual(prepare._verify_graph_alignment(saved, path, load_manifest(path)),
                         fixture.graph_saved["dataset_fingerprint"])
        for filename in ("sample0.pt", "normalization.pt"):
            with self.subTest(changed_file=filename):
                original = (relocated / filename).read_bytes()
                (relocated / filename).write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, "Graph dataset fingerprint differs"):
                    prepare._verify_graph_alignment(saved, path, load_manifest(path))
                (relocated / filename).write_bytes(original)
        saved["split_ids"]["validation"] = ["sample3"]
        with self.assertRaisesRegex(ValueError, "Graph checkpoint split assignments differ"):
            prepare._verify_graph_alignment(saved, path, load_manifest(path))

    def test_legacy_graph_checkpoint_also_requires_identical_caches(self):
        fixture = self.fixture
        relocated = self.root / "relocated"
        shutil.copytree(fixture.graph_dir, relocated)
        path = relocated / "manifest.json"
        saved = copy.deepcopy(fixture.graph_saved)
        saved.pop("dataset_fingerprint")
        self.assertEqual(prepare._verify_graph_alignment(saved, path, load_manifest(path)),
                         fixture.graph_saved["dataset_fingerprint"])
        with (relocated / "sample0.pt").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Graph cached features differ"):
            prepare._verify_graph_alignment(saved, path, load_manifest(path))

    def test_independent_training_superset_preserves_paired_rows(self):
        fixture = self.fixture
        provenance = prepare._verify_text_alignment(fixture.text_saved, fixture.text_checkpoint, fixture.source)
        self.assertTrue(provenance["independent_labels_verified"])
        self.assertTrue(provenance["text_training_superset"])
        self.assertEqual(provenance["text_training_samples"], 2)
        self.assertEqual(provenance["paired_training_samples"], 1)
        self.assertEqual(provenance["text_dataset_sha256"], prepare.file_hash(fixture.csv_path))
        self.assertEqual(provenance["text_split_indices_sha256"],
                         prepare.file_hash(fixture.text_dir / "split_indices.json"))

    def test_independent_csv_hash_and_saved_manifest_are_required(self):
        fixture = self.fixture
        saved = copy.deepcopy(fixture.text_saved)
        saved.pop("dataset_sha256")
        with self.assertRaisesRegex(ValueError, "independent-label provenance differs"):
            prepare._verify_text_alignment(saved, fixture.text_checkpoint, fixture.source)
        saved_manifest = copy.deepcopy(fixture.prepared)
        saved_manifest["train_support"][fixture.labels[0]] = 0
        write_json(fixture.text_dir / "dataset_manifest.json", saved_manifest)
        with self.assertRaisesRegex(ValueError, "independent-label provenance differs"):
            prepare._verify_text_alignment(fixture.text_saved, fixture.text_checkpoint, fixture.source)
        with fixture.csv_path.open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "training CSV hash differs"):
            prepare._verify_text_alignment(fixture.text_saved, fixture.text_checkpoint, fixture.source)

    def test_independent_split_partition_and_paired_labels_are_verified(self):
        fixture = self.fixture
        for replacement in ({"train": [0, 1], "validation": [2], "test": [2]},
                            {"train": [0, 1], "validation": [3], "test": [2]}):
            with self.subTest(splits=replacement):
                write_json(fixture.text_dir / "split_indices.json", replacement)
                with self.assertRaisesRegex(ValueError, "splits must|independent-label provenance differs"):
                    prepare._verify_text_alignment(fixture.text_saved, fixture.text_checkpoint, fixture.source)
        write_json(fixture.text_dir / "split_indices.json", fixture.splits)
        for change, message in (({"labels": [0, 0]}, "caption or labels differ"),
                                ({"ytid": "missing-video"}, "split assignments differ"),
                                ({"split": "test"}, "split assignments differ")):
            with self.subTest(change=change):
                source = copy.deepcopy(fixture.source)
                source["records"][0].update(change)
                with self.assertRaisesRegex(ValueError, message):
                    prepare._verify_text_alignment(fixture.text_saved, fixture.text_checkpoint, source)
        source = copy.deepcopy(fixture.source)
        source["label_names"].reverse()
        with self.assertRaisesRegex(ValueError, "CSV label order differs"):
            prepare._verify_text_alignment(fixture.text_saved, fixture.text_checkpoint, source)

    def test_memory_mapped_extraction_keeps_alignment_and_publishes_atomically(self):
        fixture = self.fixture
        output = self.root / "features.pt"
        with patch.object(prepare.AutoConfig, "from_pretrained", return_value=fixture.config), \
                patch.object(prepare.AutoTokenizer, "from_pretrained", return_value=TinyTokenizer()), \
                patch.object(prepare.np, "memmap", wraps=np.memmap) as mapped, \
                patch.object(prepare.torch, "load", wraps=torch.load) as loading, \
                contextlib.redirect_stdout(io.StringIO()):
            actual = prepare.prepare_features(fixture.processed_path, fixture.source_path,
                                              fixture.text_checkpoint, fixture.graph_checkpoint,
                                              output, device="cpu", batch_size=2, cpu_threads=1)
        self.assertEqual(actual, output)
        self.assertEqual(mapped.call_count, 1)
        self.assertEqual(mapped.call_args.kwargs["shape"], (3, 4, 8))
        self.assertTrue(all(call.kwargs.get("mmap") for call in loading.call_args_list[:2]))
        self.assertFalse(list(self.root.glob("task3_extract_*")))
        data = load_features(output, mmap=True)
        self.assertEqual(data["sample_ids"], ["sample0", "sample2", "sample3"])
        self.assertEqual(data["splits"], list(prepare.SPLITS))
        self.assertEqual(data["texts"], ["caption 0", "caption 2", "caption 3"])
        self.assertEqual(data["graph"].shape, (3, 8))
        self.assertEqual(data["tokens"].dtype, torch.float32)
        self.assertEqual(data["token_strings"][1], ["token_1", "token_6", "token_2", "token_0"])
        torch.testing.assert_close(data["labels"], torch.tensor([[1., 0.], [1., 1.], [0., 0.]]))
        torch.testing.assert_close(data["mask"], torch.tensor([[True, True, True, False]]).expand(3, 4))
        inputs = TinyTokenizer()(data["texts"], truncation=True, padding="max_length", max_length=4, return_tensors="pt")
        with torch.inference_mode():
            torch.testing.assert_close(data["tokens"], fixture.text_model(**inputs).last_hidden_state)
        self.assertTrue(data["provenance"]["independent_labels_verified"])
        self.assertTrue(data["provenance"]["frozen_encoders"])
        self.assertEqual(data["provenance"]["graph_dataset_fingerprint"], fixture.graph_saved["dataset_fingerprint"])
        del data
        before = output.read_bytes()
        with patch.object(prepare, "load_manifest", side_effect=AssertionError("Must reject before reading inputs")):
            with self.assertRaises(FileExistsError):
                prepare.prepare_features("missing", "missing", "missing", "missing", output)
        self.assertEqual(output.read_bytes(), before)

    def test_concurrent_output_is_preserved_and_temporary_mapping_is_cleaned(self):
        fixture = self.fixture
        output = self.root / "features.pt"
        link = prepare.os.link

        def publish_after_competing_writer(source, destination):
            Path(destination).write_bytes(b"concurrent writer")
            return link(source, destination)

        with patch.object(prepare.AutoConfig, "from_pretrained", return_value=fixture.config), \
                patch.object(prepare.AutoTokenizer, "from_pretrained", return_value=TinyTokenizer()), \
                patch.object(prepare.os, "link", side_effect=publish_after_competing_writer), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(FileExistsError):
                prepare.prepare_features(fixture.processed_path, fixture.source_path,
                                         fixture.text_checkpoint, fixture.graph_checkpoint,
                                         output, device="cpu", batch_size=2, cpu_threads=1)
        self.assertEqual(output.read_bytes(), b"concurrent writer")
        self.assertFalse(list(self.root.glob("task3_extract_*")))


if __name__ == "__main__":
    unittest.main()
