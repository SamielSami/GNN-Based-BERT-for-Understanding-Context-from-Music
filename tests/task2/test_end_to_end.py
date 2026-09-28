"""Synthetic audio smoke test for preprocessing, GraphSAGE training, and outputs."""
from __future__ import annotations

import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.task2.dataset import ProcessedTask2Dataset
from src.task2.features import AudioFeatureConfig
from src.task2.manifest import build_musiccaps_manifest
from src.task2.preprocess import preprocess_manifest
from src.task2.train import Task2TrainConfig, train_task2


class Task2EndToEndTests(unittest.TestCase):
    def test_musiccaps_manifest_preserves_label_names_with_spaces(self):
        metadata = pd.DataFrame(
            {
                "ytid": [f"video_{index}" for index in range(7)],
                "start_s": [0] * 7,
                "end_s": [10] * 7,
                "caption": [f"caption {index}" for index in range(7)],
            }
        )
        labels = pd.DataFrame(
            {
                "text": metadata["caption"],
                "male vocals": [index % 2 for index in range(7)],
                "r&b": [(index + 1) % 2 for index in range(7)],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_musiccaps_manifest(
                metadata,
                labels,
                audio_dir=Path(directory),
                include_missing=True,
            )
        self.assertEqual(manifest["label_names"], ["male vocals", "r&b"])
        self.assertEqual(manifest["records"][1]["labels"], [1, 0])

    def test_synthetic_preprocess_and_one_epoch_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            records = []
            split_names = ["train"] * 6 + ["validation", "test"]
            sample_rate = 8_000
            time = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
            for index, split in enumerate(split_names):
                labels = [index % 2, (index // 2) % 2]
                frequency = 220 + 110 * labels[0] + 55 * labels[1]
                waveform = (0.5 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
                path = audio_dir / f"sample_{index}.wav"
                sf.write(path, waveform, sample_rate)
                records.append(
                    {
                        "sample_id": f"sample_{index}",
                        "audio_path": str(path),
                        "labels": labels,
                        "split": split,
                    }
                )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "label_names": ["low_tone", "high_tone"],
                        "records": records,
                    }
                ),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                processed_manifest = preprocess_manifest(
                    manifest_path,
                    root / "processed",
                    config=AudioFeatureConfig(
                        sample_rate=sample_rate,
                        segment_seconds=0.25,
                        n_mels=16,
                        n_mfcc=8,
                        n_fft=256,
                        hop_length=64,
                    ),
                    top_k=1,
                    seed=42,
                )
            train_dataset = ProcessedTask2Dataset(processed_manifest, "train")
            self.assertEqual(len(train_dataset), 6)
            self.assertEqual(train_dataset[0]["mel"].shape[0], 1)
            output_dir = root / "results"
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = train_task2(
                    Task2TrainConfig(
                        manifest=str(processed_manifest),
                        output_dir=str(output_dir),
                        model="gnn",
                        hidden_dim=8,
                        layers=1,
                        batch_size=3,
                        epochs=1,
                        patience=1,
                        device="cpu",
                    )
                )
            self.assertEqual(metrics["model"], "gnn")
            self.assertEqual(metrics["test"]["samples"], 1)
            for filename in (
                "best_model.pt",
                "metrics.json",
                "training_curves.png",
                "test_predictions.npz",
                "test_sample_ids.json",
            ):
                self.assertTrue((output_dir / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
