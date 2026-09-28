"""Extract aligned frozen BERT tokens and GraphSAGE embeddings from trained checkpoints."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.task1.audioset import LABEL_SOURCE, load_prepared_manifest
from src.task1.train import load_tag_data
from src.task2.dataset import ProcessedTask2Dataset, dataset_identity, load_manifest, resolve_record_path
from src.task2.models import build_task2_model
from src.task2.train import resolve_device


SPLITS = ("train", "validation", "test")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_text_alignment(saved, checkpoint, source):
    """Permit a training superset only when every paired example preserves its partition."""
    config = saved["config"]
    csv_path = Path(config["data_path"])
    csv_hash = file_hash(csv_path)
    if saved.get("dataset_sha256") and saved["dataset_sha256"] != csv_hash:
        raise ValueError("Task 1 training CSV hash differs from its checkpoint")
    texts, targets, tags = load_tag_data(csv_path, config.get("text_col", "text"))
    if tags != source["label_names"]:
        raise ValueError("Task 1 training CSV label order differs")
    split_path = Path(checkpoint).with_name("split_indices.json")
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    flat = [i for indices in splits.values() for i in indices]
    if (set(splits) != set(SPLITS) or not all(splits.values()) or
            any(type(i) is not int for i in flat) or sorted(flat) != list(range(len(texts)))):
        raise ValueError("Task 1 splits must be nonempty, disjoint and cover its training CSV")
    row_splits = {i: split for split, indices in splits.items() for i in indices}
    by_text = {text: i for i, text in enumerate(texts)}
    frame = pd.read_csv(csv_path, dtype={"ytid": str})
    by_id = None
    if "ytid" in frame:
        if frame.ytid.duplicated().any():
            raise ValueError("Task 1 CSV contains duplicate video IDs")
        by_id = {value: i for i, value in enumerate(frame.ytid)}
    independent = saved.get("label_source") == LABEL_SOURCE
    if independent:
        prepared = load_prepared_manifest(csv_path)
        saved_manifest_path = Path(checkpoint).with_name("dataset_manifest.json")
        saved_manifest = json.loads(saved_manifest_path.read_text(encoding="utf-8"))
        if (prepared is None or prepared != saved_manifest or prepared["split_indices"] != splits or
                saved.get("dataset_sha256") != csv_hash):
            raise ValueError("Task 1 independent-label provenance differs from its saved training run")
        if source.get("label_source") != LABEL_SOURCE or by_id is None:
            raise ValueError("Independent AudioSet labels require aligned source label provenance and IDs")
    training_texts = {texts[i] for i in splits["train"]}
    heldout_texts = {r["text"].strip() for r in source["records"] if r["split"] != "train"}
    if training_texts & heldout_texts:
        raise ValueError("Task 1 encoder trained on Task 3 held-out captions; use aligned encoder splits")
    for row in source["records"]:
        if by_id is not None and row.get("ytid"):
            index = by_id.get(row["ytid"])
        elif independent:
            raise ValueError("Independent-label paired rows require video IDs")
        else:
            index = by_text.get(row["text"].strip())
        if index is None or row_splits[index] != row["split"]:
            raise ValueError("Task 1 caption/ID split assignments differ; use aligned encoder splits")
        if texts[index] != row["text"].strip() or targets[index].tolist() != row["labels"]:
            raise ValueError("Task 1 caption or labels differ from the paired source")
    paired_train = sum(r["split"] == "train" for r in source["records"])
    return {
        "label_source": saved.get("label_source", "legacy_unverified"),
        "independent_labels_verified": independent,
        "text_dataset_sha256": csv_hash,
        "text_split_indices_sha256": file_hash(split_path),
        "text_training_samples": len(splits["train"]),
        "paired_training_samples": paired_train,
        "text_training_superset": len(splits["train"]) > paired_train,
        "text_training_scope": "Task 1 training partition; paired rows preserve video/caption splits",
    }


def _verify_graph_alignment(saved, processed_manifest, processed):
    config = saved["config"]
    if config["model"] != "gnn":
        raise ValueError("The audio checkpoint must be a trained GraphSAGE model")
    identity = dataset_identity(processed_manifest)
    fingerprint = saved.get("dataset_fingerprint")
    if fingerprint:
        if identity["sha256"] != fingerprint:
            raise ValueError("Graph dataset fingerprint differs from its encoder training run")
        # The fingerprint verifies cached tensors and normalization after path relocation.
        if saved.get("split_ids"):
            current = {s: {r["sample_id"] for r in processed["records"] if r["split"] == s} for s in SPLITS}
            if current != {s: set(ids) for s, ids in saved["split_ids"].items()}:
                raise ValueError("Graph checkpoint split assignments differ")
    else:
        training_path = Path(config["manifest"])
        training = load_manifest(training_path)
        current = {r["sample_id"]: r["split"] for r in processed["records"]}
        if current != {r["sample_id"]: r["split"] for r in training["records"]}:
            raise ValueError("Graph checkpoint split assignments differ; retrain on the aligned split")
        if processed.get("feature_config") != training.get("feature_config"):
            raise ValueError("Graph preprocessing feature configurations differ")
        if any(not m.get("normalization_path") for m in (processed, training)):
            raise ValueError("Processed manifests must reference training normalization")
        if file_hash(resolve_record_path(processed_manifest, processed["normalization_path"])) != file_hash(
                resolve_record_path(training_path, training["normalization_path"])):
            raise ValueError("Graph normalization differs from the encoder training run")
        if dataset_identity(training_path)["sha256"] != identity["sha256"]:
            raise ValueError("Graph cached features differ from the encoder training manifest")
    return identity["sha256"]


def _load_text_encoder(saved, device):
    # Attach mmap-backed checkpoint weights to a meta model without loading a second encoder.
    config = AutoConfig.from_pretrained(saved["config"]["model_name"])
    with torch.device("meta"):
        model = AutoModel.from_config(config)
    state = {key.removeprefix("bert."): value for key, value in saved["state_dict"].items()
             if key.startswith("bert.")}
    model.load_state_dict(state, strict=True, assign=True)
    # BERT-family position/token-type IDs are nonpersistent buffers, absent from checkpoints.
    for module in model.modules():
        for name, value in module.named_buffers(recurse=False):
            if value.is_meta:
                if name == "position_ids":
                    module.register_buffer(name, torch.arange(value.shape[-1]).expand(value.shape), persistent=False)
                elif name == "token_type_ids":
                    module.register_buffer(name, torch.zeros(value.shape, dtype=value.dtype), persistent=False)
                else:
                    raise ValueError(f"Unsupported nonpersistent encoder buffer: {name}")
    return model.to(device).eval().requires_grad_(False)


def prepare_features(processed_manifest, source_manifest, text_checkpoint, graph_checkpoint,
                     output, device="auto", batch_size=4, cpu_threads=6):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new feature output path: {output}")
    if batch_size < 1 or cpu_threads < 1:
        raise ValueError("batch_size and cpu_threads must be positive")
    torch.set_num_threads(cpu_threads)
    processed_manifest, source_manifest = Path(processed_manifest), Path(source_manifest)
    processed, source = load_manifest(processed_manifest), load_manifest(source_manifest)
    if processed["label_names"] != source["label_names"]:
        raise ValueError("Source and processed label orders differ")
    if processed.get("label_source") != source.get("label_source"):
        raise ValueError("Source and processed label provenance differs")
    source_rows = {r["sample_id"]: r for r in source["records"]}
    if set(source_rows) != {r["sample_id"] for r in processed["records"]}:
        raise ValueError("Source and processed sample IDs differ")
    if {r["split"] for r in processed["records"]} != set(SPLITS):
        raise ValueError("Require nonempty train, validation and test partitions")
    for record in processed["records"]:
        raw = source_rows[record["sample_id"]]
        if record["split"] != raw["split"] or not raw.get("text", "").strip():
            raise ValueError("Each graph requires a caption and matching split")
        if "labels" not in raw or ("labels" in record and record["labels"] != raw["labels"]):
            raise ValueError("Source and processed labels differ")
        if record.get("ytid") != raw.get("ytid"):
            raise ValueError("Source and processed video IDs differ")

    text_saved = torch.load(text_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    graph_saved = torch.load(graph_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if "config" not in text_saved:
        raise ValueError("Task 3 requires a metadata-rich Task 1 checkpoint with split provenance")
    labels = processed["label_names"]
    if text_saved["tags"] != labels or graph_saved["label_names"] != labels:
        raise ValueError("Checkpoint label names/order must match the paired manifest")
    fingerprint = _verify_graph_alignment(graph_saved, processed_manifest, processed)
    provenance = _verify_text_alignment(text_saved, text_checkpoint, source)
    gc, tc = graph_saved["config"], text_saved["config"]
    provenance.update(
        synthetic=bool(processed.get("synthetic", False) or graph_saved.get("synthetic", False)),
        text_checkpoint_sha256=file_hash(text_checkpoint), graph_checkpoint_sha256=file_hash(graph_checkpoint),
        processed_manifest_sha256=file_hash(processed_manifest), source_manifest_sha256=file_hash(source_manifest),
        graph_dataset_fingerprint=fingerprint, graph_variant=gc["graph_variant"],
        tokenizer_name=tc["model_name"], max_length=tc["max_length"],
        frozen_encoders=True, token_dtype="float32", batch_size=batch_size, cpu_threads=cpu_threads,
    )
    if not provenance["independent_labels_verified"]:
        provenance["warning"] = "Legacy label provenance is unverified; caption-derived proxies may leak targets"

    device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(tc["model_name"])
    text_model = _load_text_encoder(text_saved, device)
    graph_model = build_task2_model("gnn", input_dim=graph_saved["input_dim"], num_labels=len(labels),
                                    hidden_dim=gc["hidden_dim"], layers=gc["layers"], dropout=gc["dropout"]).to(device)
    graph_model.load_state_dict(graph_saved["state_dict"])
    graph_model.eval().requires_grad_(False)
    n, length = len(source_rows), tc["max_length"]
    result = {key: [] for key in ("sample_ids", "splits", "texts", "token_strings")}
    result.update(mask=torch.empty(n, length, dtype=torch.bool), token_ids=torch.empty(n, length, dtype=torch.long),
                  graph=torch.empty(n, graph_model.encoder.output_dim), labels=torch.empty(n, len(labels)),
                  label_names=labels, provenance=provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="task3_extract_", dir=output.parent) as directory:
        mapped = np.memmap(Path(directory) / "tokens.dat", mode="w+", dtype=np.float32,
                           shape=(n, length, text_model.config.hidden_size))
        try:
            offset = 0
            with torch.inference_mode():
                for split in SPLITS:
                    dataset = ProcessedTask2Dataset(processed_manifest, split, gc["graph_variant"])
                    for start in range(0, len(dataset), batch_size):
                        samples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
                        rows = [source_rows[sample["sample_id"]] for sample in samples]
                        for sample, row in zip(samples, rows):
                            if sample["labels"].tolist() != row["labels"]:
                                raise ValueError("Cached labels differ from source; rerun Task 2 preprocessing")
                        encoded = tokenizer([row["text"] for row in rows], truncation=True, padding="max_length",
                                            max_length=length, return_tensors="pt")
                        end = offset + len(samples)
                        result["token_ids"][offset:end] = encoded["input_ids"]
                        result["mask"][offset:end] = encoded["attention_mask"].bool()
                        result["token_strings"].extend(tokenizer.convert_ids_to_tokens(ids.tolist())
                                                      for ids in encoded["input_ids"])
                        encoded = {key: value.to(device) for key, value in encoded.items()}
                        tokens = text_model(**encoded).last_hidden_state.cpu()
                        mapped[offset:end] = tokens.numpy()
                        graph = Batch.from_data_list([sample["graph"] for sample in samples]).to(device)
                        result["graph"][offset:end] = graph_model.encoder(graph).cpu()
                        result["labels"][offset:end] = torch.stack([sample["labels"] for sample in samples])
                        for row in rows:
                            result["sample_ids"].append(row["sample_id"])
                            result["splits"].append(split)
                            result["texts"].append(row["text"])
                        offset = end
                        if start == 0 or end % (batch_size * 25) == 0 or start + batch_size >= len(dataset):
                            mapped.flush()
                            print(f"extract split={split} samples={offset}/{n}", flush=True)
            if dataset_identity(processed_manifest)["sha256"] != fingerprint:
                raise ValueError("Graph dataset changed during extraction")
            result["tokens"] = torch.from_numpy(mapped)
            temporary_output = Path(directory) / "features.pt"
            torch.save(result, temporary_output)
            # Same-filesystem hard link publishes atomically and refuses an existing output.
            os.link(temporary_output, output)
        finally:
            result.pop("tokens", None)
            mapped.flush()
            mapped._mmap.close()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("processed-manifest", "source-manifest", "text-checkpoint", "graph-checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=6)
    print(prepare_features(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
