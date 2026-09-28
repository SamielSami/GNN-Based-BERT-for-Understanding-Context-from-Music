"""Create retrieval comparison plots and caption-based diagnostics from saved runs."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiment import DIRECTIONS, read, verify, write


def analyze(output_dir, features=None):
    root = Path(output_dir).resolve()
    verification = verify(root, features)
    selection = read(root / "selection.json")
    split = "test" if verification["evaluated_test"] else "validation"
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    frame = pd.read_csv(root / "comparison.csv")
    aggregate = pd.read_csv(root / "comparison_aggregate.csv")
    n = int(frame.loc[frame.split == split, "gallery_size"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = np.arange(3)
    for ax, direction in zip(axes, DIRECTIONS):
        for j, (baseline, color) in enumerate((("trained", "#1565a0"), ("untrained", "#8b939b"))):
            row = aggregate[(aggregate.split == split) & (aggregate.baseline == baseline) & (aggregate.direction == direction)].iloc[0]
            ax.bar(x + (j - 0.5) * 0.26, [row[f"recall_at_{k}_mean"] * 100 for k in (1, 5, 10)],
                   width=0.26, yerr=[np.nan_to_num(row[f"recall_at_{k}_std"]) * 100 for k in (1, 5, 10)],
                   capsize=3, label=baseline.capitalize(), color=color)
        ax.scatter(x, [min(k, n) / n * 100 for k in (1, 5, 10)], marker="_", s=230,
                   color="#c04c18", label="Random expectation", zorder=5)
        ax.set(xticks=x, xticklabels=["R@1", "R@5", "R@10"], title=direction.replace("_", " "), ylabel="Recall (%)")
        ax.legend(fontsize=8)
    fig.suptitle(f"Held-out retrieval ({n} pairs)" if split == "test" else f"Validation retrieval ({n} pairs)")
    fig.tight_layout()
    fig.savefig(output / "retrieval_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run in selection["runs"]:
        history = read(root / f"seed{run['seed']}/history.json")
        epochs = [r["epoch"] for r in history]
        axes[0].plot(epochs, [r["train_loss"] for r in history], label=f"Seed {run['seed']}")
        axes[1].plot(epochs, [r["validation"]["mean_recall_at_1"] * 100 for r in history], label=f"Seed {run['seed']}")
        axes[1].scatter(run["best_epoch"], run["validation_mean_recall_at_1"] * 100, marker="*", s=100)
    axes[0].set(xlabel="Epoch", ylabel="Training symmetric InfoNCE")
    axes[1].set(xlabel="Epoch", ylabel="Validation mean Recall@1 (%)")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)

    # Use the first declared seed, never the seed with the best test result.
    seed = selection["seeds"][0]
    examples = read(root / f"seed{seed}/{split}/examples.json")
    cases = [f"# Retrieval diagnostics: seed {seed}, {split}", "",
             "Three fixed queries per direction plus best/worst-rank diagnostics. "
             "Outcome-selected examples illustrate extremes, not typical quality. "
             "Captions below are dataset annotations; no human listening evaluation was performed.", ""]
    for example in examples:
        cases += [f"## {example['direction']}: {example['query_id']}", "",
                  f"Selection: `{example['selection']}`. Paired rank: **{example['paired_rank']} / {n}**.", "",
                  "Query/paired caption: " + " ".join(example["caption"].split()), ""]
        for match in example["ranked_matches"][:3]:
            cases.append(f"- Rank {match['rank']}: `{match['sample_id']}`; cosine {match['score']:.4f}; "
                         f"{'paired positive' if match['is_paired_positive'] else 'nonmatching ID'}. "
                         + " ".join(match["caption"].split()))
        cases.append("")
    (output / "cases.md").write_text("\n".join(cases), encoding="utf-8")

    rank_summary = []
    for run in selection["runs"]:
        ranks = read(root / f"seed{run['seed']}/{split}/ranks.json")
        for direction in DIRECTIONS:
            values = np.asarray(ranks[direction])
            rank_summary.append(dict(seed=run["seed"], direction=direction, queries=len(values),
                                     paired_top1=int((values == 1).sum()), paired_top10=int((values <= 10).sum()),
                                     paired_outside_top10=int((values > 10).sum()), worst_rank=int(values.max())))
    write(output / "rank_summary.json", rank_summary)
    return dict(split=split, seed_for_cases=seed, output_dir=str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--features")
    args = parser.parse_args()
    print(analyze(args.output_dir, args.features))
