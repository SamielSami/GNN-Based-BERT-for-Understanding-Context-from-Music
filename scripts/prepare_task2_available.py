"""Freeze an aligned Task 2 subset after validating downloaded audio files."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.task2.manifest import build_aligned_musiccaps_manifest


def validate_audio(path, duration):
    """Decode the entire clip, checking duration, finite samples and concurrent edits."""
    path = Path(path)
    before = path.stat()
    frames = 0
    with sf.SoundFile(path) as audio:
        rate = audio.samplerate
        for block in audio.blocks(blocksize=65536, dtype='float32'):
            if not np.isfinite(block).all():
                raise ValueError('non-finite audio samples')
            frames += len(block)
    if frames == 0 or abs(frames / rate - duration) > .02:
        raise ValueError(f'duration {frames / rate:.4f}s; expected {duration:g}s')
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('file changed during validation; pause the downloader')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir', type=Path, default=Path('data/raw/musiccaps_audio'))
    parser.add_argument('--output', type=Path, default=Path('data/splits/task2_available.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Frozen manifest already exists: {args.output}. Reuse it or choose a new --output.')
    metadata = Path('data/raw/musiccaps_official.csv')
    labels = Path('data/processed/musiccaps_audioset.csv')
    source_path = Path('results/task1/audioset_cpu_20260907/dataset_manifest.json')
    source = json.loads(source_path.read_text(encoding='utf-8'))
    if hashlib.sha256(labels.read_bytes()).hexdigest() != source['csv_sha256']:
        raise ValueError('Task 1 label CSV checksum mismatch')
    manifest = build_aligned_musiccaps_manifest(
        pd.read_csv(metadata, dtype={'ytid': str}), pd.read_csv(labels, dtype={'ytid': str}),
        task1_manifest=source, audio_dir=args.audio_dir, audio_mode='pretrimmed',
    )
    retained, excluded = [], []
    for i, record in enumerate(manifest['records'], 1):
        try:
            validate_audio(record['audio_path'], record['audio_duration_seconds'])
            retained.append(record)
        except (OSError, ValueError, RuntimeError) as error:
            excluded.append({'sample_id': record['sample_id'], 'reason': str(error)})
        if i % 250 == 0:
            print(f'Audio checked: {i}/{len(manifest["records"])}', flush=True)
    counts = Counter(record['split'] for record in retained)
    summary = {'valid_clips': len(retained), 'split_counts': dict(counts), 'excluded_invalid_audio': excluded,
               'validation': 'Full decoded audio; finite samples; duration within 20 ms; unchanged file stat'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.validation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if len(retained) < 20 or any(counts[s] == 0 for s in ('train', 'validation', 'test')):
        raise ValueError('Need at least 20 valid clips and nonempty train, validation and test splits')
    manifest['records'] = retained
    manifest['audit'].update(split_counts=dict(counts), ready_for_training=True,
                             valid_audio=len(retained), invalid_audio=len(excluded),
                             excluded_invalid_audio=excluded)
    manifest['audit']['label_support_by_split'] = {
        split: {tag: sum(r['labels'][i] for r in retained if r['split'] == split)
                for i, tag in enumerate(manifest['label_names'])}
        for split in ('train', 'validation', 'test')
    }
    manifest['task1_alignment'].update(source_manifest=str(source_path.resolve()),
        source_manifest_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        source_csv=str(labels.resolve()), source_csv_hash_verified=True)
    manifest['audio_validation'] = summary['validation']
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps({'manifest': str(args.output), 'valid_clips': len(retained),
                      'split_counts': dict(counts), 'excluded_invalid_clips': len(excluded)}, indent=2))


if __name__ == '__main__':
    main()
