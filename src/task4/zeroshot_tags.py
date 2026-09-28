"""Training-free caption/tag prompt similarity versus saved supervised Task 3."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.task2.train import classification_metrics
from src.task3.data import load_features, feature_fingerprint
from .experiment import read, write
from .train import empty_output
from .zeroshot import MODEL, REVISION, pooled


def tag_scores(captions, positive, negative):
    """Fixed cosine margin; sigmoid is a bounded score, not calibrated probability."""
    margin = captions @ positive.T - captions @ negative.T
    return 1 / (1 + np.exp(-margin))


def run(features, clap_dir, task3_dir, output_dir):
    from transformers import ClapModel, ClapProcessor
    torch.set_num_threads(6)
    data = load_features(features, mmap=True)
    indices = [i for i,s in enumerate(data["splits"]) if s=="test"]
    ids = np.asarray([data["sample_ids"][i] for i in indices])
    targets = data["labels"][indices].numpy()
    names = read(Path(__file__).resolve().parents[1] / "task1/audioset_names.json")
    labels = data["label_names"]
    positive = [f"This music contains {names.get(tag, tag)}." for tag in labels]
    negative = [f"This music does not contain {names.get(tag, tag)}." for tag in labels]
    root = empty_output(output_dir)
    clap = Path(clap_dir)
    config = read(clap / "config.json")
    if config["model"] != MODEL or config["revision"] != REVISION or config["feature_sha256"] != feature_fingerprint(features):
        raise ValueError("CLAP provenance differs")
    with np.load(clap / "embeddings.npz",allow_pickle=False) as saved:
        if not np.array_equal(saved["sample_ids"],ids):
            raise ValueError("CLAP test IDs differ")
        captions = saved["text"]
    declaration = dict(model=MODEL,revision=REVISION,positive_prompts=positive,negative_prompts=negative,
                       threshold=0.5, score="sigmoid(cos(caption,positive_tag_prompt)-cos(caption,negative_tag_prompt))",
                       local_training=False,validation_calibration=False,prompt_selection="Fixed template; no prompt search",
                       feature_sha256=feature_fingerprint(features),caption_embeddings_sha256=feature_fingerprint(clap/'embeddings.npz'),
                       caveat="CLAP text-to-text similarity heuristic; not an NLI classifier, not CLAP audio tag classification; negation may be poorly represented")
    write(root / "config.json",declaration)
    model = ClapModel.from_pretrained(MODEL,revision=REVISION).eval().requires_grad_(False)
    processor = ClapProcessor.from_pretrained(MODEL,revision=REVISION)
    with torch.inference_mode():
        inputs = processor(text=positive+negative,padding=True,truncation=True,max_length=512,return_tensors="pt")
        prompts = pooled(model.get_text_features(**inputs))
    scores = tag_scores(captions,prompts[:len(labels)],prompts[len(labels):])
    metrics = classification_metrics(targets,scores,labels,0.5)
    np.savez_compressed(root/'predictions.npz',sample_ids=ids,targets=targets,probabilities=scores,
                        label_names=np.asarray(labels),positive_embeddings=prompts[:len(labels)],negative_embeddings=prompts[len(labels):])
    write(root/'metrics.json',metrics)
    comparison = [dict(method="zero_shot_caption_prompt_similarity",seed="none",**{k:metrics[k] for k in ("macro_f1","micro_f1","mean_average_precision")})]
    supervised = Path(task3_dir)
    selection = read(supervised/'selection.json')
    mode = selection['selected_fusion']
    for seed in selection['seeds']:
        directory = supervised/f'seed{seed}'
        report = read(directory/f'{mode}_test_metrics.json')
        with np.load(directory/f'{mode}_test_predictions.npz',allow_pickle=False) as saved:
            if not np.array_equal(saved['sample_ids'],ids) or not np.array_equal(saved['targets'],targets):
                raise ValueError('Supervised comparison IDs/targets differ')
            measured = classification_metrics(targets,saved['probabilities'],labels,report['metrics']['threshold'])
        if measured != report['metrics']:
            raise ValueError('Supervised metrics differ from saved predictions')
        comparison.append(dict(method='task3_supervised_'+mode,seed=seed,**{k:measured[k] for k in ('macro_f1','micro_f1','mean_average_precision')}))
    pd.DataFrame(comparison).to_csv(root/'comparison.csv',index=False)
    with np.load(root/'predictions.npz',allow_pickle=False) as saved:
        assert classification_metrics(saved['targets'],saved['probabilities'],labels,0.5)==metrics
    write(root/'verification.json',dict(valid=True,identical_test_ids_and_targets=True,supervised_metrics_recomputed=True,
          zero_shot_metrics_recomputed=True,task3_selected_fusion=mode,
          artifacts={p.name:feature_fingerprint(p) for p in root.iterdir() if p.is_file()}))
    return comparison


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',default='data/processed/task3_available/features.pt')
    parser.add_argument('--clap-dir',default='results/task4/clap_zeroshot')
    parser.add_argument('--task3-dir',default='results/task3/available_run1')
    parser.add_argument('--output-dir',required=True)
    print(run(**vars(parser.parse_args())))
