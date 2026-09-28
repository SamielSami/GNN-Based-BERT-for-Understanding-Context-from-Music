"""Pinned pretrained CLAP comparison on the unchanged Task 4 test gallery."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.task2.features import load_audio
from src.task3.data import load_features, feature_fingerprint
from .experiment import read, write
from .metrics import retrieval_metrics, recompute
from .train import empty_output

MODEL = "laion/clap-htsat-unfused"
REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"


def aligned_records(features, manifest, split="test"):
    data = load_features(features, mmap=True)
    source = read(manifest)
    expected = data.get("provenance", {}).get("source_manifest_sha256")
    if expected and feature_fingerprint(manifest) != expected:
        raise ValueError("Audio manifest differs from the frozen feature provenance")
    records = {r["sample_id"]: r for r in source["records"]}
    if len(records) != len(source["records"]):
        raise ValueError("Duplicate audio IDs")
    rows = []
    for i, sid in enumerate(data["sample_ids"]):
        if data["splits"][i] != split:
            continue
        record = records[sid]
        if record["split"] != split or record["text"] != data["texts"][i]:
            raise ValueError("Caption/split differs from retrieval cache")
        if not Path(record["audio_path"]).is_file():
            raise FileNotFoundError(record["audio_path"])
        rows.append(record)
    return rows


def pooled(output):
    # Transformers 5 returns a model output; older releases returned tensors.
    value = output if isinstance(output, torch.Tensor) else output.pooler_output
    return F.normalize(value, dim=-1).detach().cpu().numpy()


def run(features, manifest, output_dir, batch_size=4, cpu_threads=6):
    from transformers import ClapModel, ClapProcessor
    torch.set_num_threads(cpu_threads)
    torch.manual_seed(42)
    np.random.seed(42)
    rows = aligned_records(features, manifest)
    root = empty_output(output_dir)
    config = dict(model=MODEL, revision=REVISION, split="test", sample_ids=[r["sample_id"] for r in rows],
                  feature_sha256=feature_fingerprint(features), manifest_sha256=feature_fingerprint(manifest),
                  sample_rate=48000, audio="Same local intervals as graph features; mono, resampled, peak-normalized",
                  batch_size=batch_size, cpu_threads=cpu_threads, text_max_length=512,
                  task_specific_training=False, checkpoint_selection="Fixed before inference; no local tuning",
                  scope="Text/audio CLAP comparator; audio embeddings are not graph embeddings",
                  pretraining_overlap="Not established: zero-shot means no training on this project's split, not certified unseen pretraining audio")
    write(root / "config.json", config)
    model = ClapModel.from_pretrained(MODEL, revision=REVISION).eval().requires_grad_(False)
    processor = ClapProcessor.from_pretrained(MODEL, revision=REVISION)
    texts, audios, hashes = [], [], {}
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            waves = []
            for row in batch:
                waves.append(load_audio(row["audio_path"], 48000,
                                        offset_seconds=row["audio_offset_seconds"],
                                        duration_seconds=row["audio_duration_seconds"]))
                hashes[row["sample_id"]] = feature_fingerprint(row["audio_path"])
            audio_input = processor(audio=waves, sampling_rate=48000, return_tensors="pt")
            text_input = processor(text=[r["text"] for r in batch], return_tensors="pt", padding=True,
                                   truncation=True, max_length=512)
            audios.append(pooled(model.get_audio_features(**audio_input)))
            texts.append(pooled(model.get_text_features(**text_input)))
            if start % (batch_size * 10) == 0:
                print(f"CLAP: {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    text, audio = np.concatenate(texts), np.concatenate(audios)
    np.savez_compressed(root / "embeddings.npz", text=text, graph=audio,
                        sample_ids=np.asarray(config["sample_ids"]))
    metrics = retrieval_metrics(text, audio)
    write(root / "metrics.json", dict(metrics=metrics, modality_aliases={"graph": "CLAP audio (not GNN)"},
                                     inference_seconds=time.perf_counter() - started))
    write(root / "audio_sha256.json", hashes)
    assert recompute(root / "embeddings.npz") == metrics
    write(root / "verification.json", dict(valid=True, metrics_recomputed=True,
          config_sha256=feature_fingerprint(root / "config.json"), embeddings_sha256=feature_fingerprint(root / "embeddings.npz"),
          metrics_sha256=feature_fingerprint(root / "metrics.json"), model=MODEL, revision=REVISION,
          pairs=len(rows), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/processed/task3_available/features.pt")
    parser.add_argument("--manifest", default="data/splits/task2_available.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    print(run(**vars(args)))
