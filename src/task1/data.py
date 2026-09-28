"""Build independent AudioSet targets (default) or the historical lexical proxy.

The default CLI delegates ID-first preparation to ``audioset.py``. The legacy
``extract_tags`` and ``build_tag_frame`` helpers below still implement lexical
proxy labels and are retained for historical reproduction only; they do not
implement the independent-label academic protocol.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd


TAG_VOCAB: dict[str, tuple[str, ...]] = {
    # Genre
    "rock": (r"rock",),
    "pop": (r"pop",),
    "jazz": (r"jazz",),
    "classical": (r"classical", r"orchestra(l)?"),
    "electronic": (r"electronic", r"electro"),
    "hip hop": (r"hip[\s-]?hop", r"rap"),
    "blues": (r"blues",),
    "country": (r"country",),
    "reggae": (r"reggae",),
    "metal": (r"metal",),
    "folk": (r"folk",),
    "funk": (r"funk",),
    "soul": (r"soul",),
    "ambient": (r"ambient",),
    "techno": (r"techno",),
    "house": (r"house music", r"\bhouse\b"),
    "disco": (r"disco",),
    "punk": (r"punk",),
    "latin": (r"latin",),
    "r&b": (r"r&b", r"r and b", r"rhythm and blues"),
    # Mood
    "happy": (r"happy", r"cheerful", r"joyful"),
    "sad": (r"sad", r"melancholic", r"melancholy", r"somber"),
    "energetic": (r"energetic", r"upbeat", r"lively"),
    "calm": (r"calm", r"peaceful", r"soothing", r"relax(ing|ed)?"),
    "romantic": (r"romantic",),
    "angry": (r"angry", r"aggressive"),
    "dark": (r"dark", r"eerie", r"haunting"),
    "uplifting": (r"uplifting", r"triumphant"),
    "dreamy": (r"dreamy", r"ethereal"),
    "tense": (r"tense", r"suspenseful"),
    # Instrument and production
    "piano": (r"piano",),
    "guitar": (r"guitar",),
    "drums": (r"drum(s)?", r"percussion"),
    "violin": (r"violin", r"strings"),
    "saxophone": (r"saxophone", r"\bsax\b"),
    "synthesizer": (r"synthesizer", r"synth"),
    "bass": (r"bass",),
    "vocals": (r"vocal(s)?", r"singing", r"singer"),
    "flute": (r"flute",),
    "trumpet": (r"trumpet", r"brass"),
    "acoustic": (r"acoustic",),
    "male vocals": (r"\bmale vocal", r"\bmale singer"),
    "female vocals": (r"female vocal", r"female singer"),
    "instrumental": (r"instrumental",),
    "choir": (r"choir", r"chorus"),
    "electric guitar": (r"electric guitar",),
    "live": (r"live recording", r"\blive\b"),
    "slow": (r"\bslow\b", r"slow tempo"),
    "fast": (r"\bfast\b", r"fast tempo", r"up[\s-]?tempo"),
    "loop": (r"\bloop(ing|ed)?\b",),
}

_PATTERNS = {
    tag: re.compile("|".join(patterns), re.IGNORECASE)
    for tag, patterns in TAG_VOCAB.items()
}


def extract_tags(caption: str) -> list[str]:
    """Return canonical tags whose lexical patterns occur in *caption*."""
    text = str(caption)
    return [tag for tag, pattern in _PATTERNS.items() if pattern.search(text)]


def build_tag_frame(
    raw: pd.DataFrame,
    *,
    caption_col: str = "caption",
    top_k: int = 50,
    keep_untagged: bool = False,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Convert captions into a validated text + multi-hot-label dataframe."""
    if caption_col not in raw.columns:
        raise ValueError(f"Caption column {caption_col!r} was not found")
    if not 1 <= top_k <= len(TAG_VOCAB):
        raise ValueError(f"top_k must be between 1 and {len(TAG_VOCAB)}")

    frame = raw.dropna(subset=[caption_col]).copy()
    frame[caption_col] = frame[caption_col].astype(str).str.strip()
    frame = frame[frame[caption_col].str.len() > 0]
    frame["_matched_tags"] = frame[caption_col].map(extract_tags)
    if not keep_untagged:
        frame = frame[frame["_matched_tags"].map(len) > 0]

    counts = Counter(tag for tags in frame["_matched_tags"] for tag in tags)
    tags = [tag for tag, _ in counts.most_common(top_k)]
    if not tags:
        raise ValueError("No configured tags matched any caption")

    for tag in tags:
        frame[tag] = frame["_matched_tags"].map(lambda hits, name=tag: int(name in hits))

    if not keep_untagged:
        frame = frame[frame[tags].sum(axis=1) > 0]
    metadata = ["ytid"] if "ytid" in frame.columns else []
    result = frame.rename(columns={caption_col: "text"})[[*metadata, "text", *tags]].reset_index(drop=True)
    return result, tags, {tag: counts[tag] for tag in tags}


def load_musiccaps() -> pd.DataFrame:
    """Load MusicCaps from Hugging Face (optional ``datasets`` dependency)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("Install the optional 'datasets' package to use --hf-dataset") from exc
    return load_dataset("google/MusicCaps", split="train").to_pandas()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf-dataset", action="store_true")
    source.add_argument("--input-csv", type=Path)
    parser.add_argument("--caption-col", default="caption")
    parser.add_argument("--label-source", choices=["audioset", "proxy"], default="audioset")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-train-support", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--keep-untagged", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/processed/musiccaps_audioset.csv"))
    args = parser.parse_args()

    raw = load_musiccaps() if args.hf_dataset else pd.read_csv(args.input_csv)
    if args.label_source == "audioset":
        from .audioset import save_audioset_dataset
        result, manifest = save_audioset_dataset(
            raw, args.output, caption_col=args.caption_col, top_k=args.top_k,
            min_train_support=args.min_train_support, seed=args.seed,
            val_size=args.val_size, test_size=args.test_size,
        )
        print(f"Saved {len(result)} captions with ytid and {len(manifest['tags'])} independent AudioSet labels")
        print("ID split sizes:", {key: len(value) for key, value in manifest["split_ids"].items()})
        return
    if args.output.exists():
        raise FileExistsError("Use a new output path to preserve existing data")
    result, tags, counts = build_tag_frame(
        raw,
        caption_col=args.caption_col,
        top_k=args.top_k,
        keep_untagged=args.keep_untagged,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} captions and {len(tags)} proxy tags to {args.output}")
    print("Tag frequencies:")
    for tag in tags:
        print(f"  {tag:20s} {counts[tag]}")
    print("WARNING: labels are keyword-derived from the input captions; report this as a proxy-task limitation.")


if __name__ == "__main__":
    main()
