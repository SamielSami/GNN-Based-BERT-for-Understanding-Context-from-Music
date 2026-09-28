"""Deterministic audio loading and Task 2 segment-level feature extraction."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass(frozen=True)
class AudioFeatureConfig:
    """Parameters shared by graph-node and CNN mel extraction."""

    sample_rate: int = 22_050
    segment_seconds: float = 1.0
    n_mels: int = 128
    n_mfcc: int = 20
    n_fft: int = 2_048
    hop_length: int = 512

    def validate(self) -> None:
        if self.sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if not np.isfinite(self.segment_seconds) or self.segment_seconds <= 0:
            raise ValueError("segment_seconds must be finite and positive")
        if round(self.sample_rate * self.segment_seconds) < 1:
            raise ValueError("segment_seconds must span at least one audio sample")
        if self.n_mels < 1 or self.n_mfcc < 1:
            raise ValueError("n_mels and n_mfcc must be positive")
        if self.n_mfcc > self.n_mels:
            raise ValueError("n_mfcc cannot exceed n_mels")
        if self.n_fft < 16 or self.hop_length < 1:
            raise ValueError("n_fft must be at least 16 and hop_length must be positive")

    @property
    def node_feature_dim(self) -> int:
        # log-mel mean/std + chroma + MFCC/delta mean + RMS/centroid/ZCR
        return 2 * self.n_mels + 12 + 2 * self.n_mfcc + 3

    def to_dict(self) -> dict:
        return asdict(self)


def load_audio(
    path: str | Path,
    sample_rate: int = 22_050,
    *,
    offset_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> np.ndarray:
    """Load a specified mono audio interval, resample, and peak-normalize.

    Offsets apply to the local file, so already trimmed MusicCaps files use zero.
    Reject incomplete intervals (allowing 20 ms of codec rounding).
    """
    if sample_rate < 1:
        raise ValueError("sample_rate must be positive")
    if not np.isfinite(offset_seconds) or offset_seconds < 0:
        raise ValueError("offset_seconds must be finite and non-negative")
    if duration_seconds is not None and (
        not np.isfinite(duration_seconds) or duration_seconds <= 0
    ):
        raise ValueError("duration_seconds must be finite and positive")
    waveform, _ = librosa.load(
        str(path), sr=sample_rate, mono=True, offset=offset_seconds, duration=duration_seconds
    )
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"Audio file contains NaN or infinity: {path}")
    if duration_seconds is not None:
        expected = round(duration_seconds * sample_rate)
        tolerance = max(1, round(0.02 * sample_rate))
        if len(waveform) < expected - tolerance:
            raise ValueError(
                f"Audio interval is too short: requested {duration_seconds:g}s at "
                f"offset {offset_seconds:g}s, loaded {len(waveform) / sample_rate:g}s: {path}"
            )
    peak = float(np.max(np.abs(waveform)))
    return waveform / peak if peak > 0 else waveform


def _validate_waveform(waveform: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("waveform must be a non-empty mono array")
    if not np.isfinite(waveform).all():
        raise ValueError("waveform contains NaN or infinity")
    return waveform


def split_segments(waveform: np.ndarray, config: AudioFeatureConfig) -> list[np.ndarray]:
    """Split into fixed windows and zero-pad the final segment."""
    config.validate()
    waveform = _validate_waveform(waveform)
    window = int(round(config.sample_rate * config.segment_seconds))
    segments = [waveform[start : start + window] for start in range(0, len(waveform), window)]
    return [
        np.pad(segment, (0, window - len(segment))).astype(np.float32, copy=False)
        for segment in segments
    ]


def _power_to_log_mel(power: np.ndarray) -> np.ndarray:
    if float(np.max(power)) <= 1e-10:
        return np.full_like(power, -80.0)
    reference = max(float(np.max(power)), 1e-10)
    return librosa.power_to_db(power, ref=reference, top_db=80.0)


def _mfcc_delta(mfcc: np.ndarray) -> np.ndarray:
    frames = mfcc.shape[1]
    width = min(9, frames if frames % 2 == 1 else frames - 1)
    if width < 3:
        return np.zeros_like(mfcc)
    return librosa.feature.delta(mfcc, width=width, mode="nearest")


def extract_segment_features(segment: np.ndarray, config: AudioFeatureConfig) -> np.ndarray:
    """Return one finite feature vector for a fixed audio segment."""
    config.validate()
    segment = _validate_waveform(segment)
    mel_power = librosa.feature.melspectrogram(
        y=segment,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        power=2.0,
    )
    log_mel = _power_to_log_mel(mel_power)
    chroma = librosa.feature.chroma_stft(
        y=segment,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=config.n_mfcc)
    delta = _mfcc_delta(mfcc)
    rms = librosa.feature.rms(y=segment, frame_length=config.n_fft, hop_length=config.hop_length)
    centroid = librosa.feature.spectral_centroid(
        y=segment,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    zcr = librosa.feature.zero_crossing_rate(
        y=segment, frame_length=config.n_fft, hop_length=config.hop_length
    )
    vector = np.concatenate(
        [
            log_mel.mean(axis=1),
            log_mel.std(axis=1),
            chroma.mean(axis=1),
            mfcc.mean(axis=1),
            delta.mean(axis=1),
            [rms.mean(), centroid.mean(), zcr.mean()],
        ]
    ).astype(np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Extracted node features contain NaN or infinity")
    if vector.shape != (config.node_feature_dim,):
        raise RuntimeError(
            f"Expected {config.node_feature_dim} node features, got {vector.shape}"
        )
    return vector


def extract_track_features_from_waveform(
    waveform: np.ndarray, config: AudioFeatureConfig
) -> np.ndarray:
    """Extract a ``[segments, feature_dim]`` node-feature matrix."""
    return np.stack(
        [extract_segment_features(segment, config) for segment in split_segments(waveform, config)]
    ).astype(np.float32)


def extract_track_features(path: str | Path, config: AudioFeatureConfig) -> np.ndarray:
    return extract_track_features_from_waveform(load_audio(path, config.sample_rate), config)


def extract_log_mel_from_waveform(
    waveform: np.ndarray, config: AudioFeatureConfig
) -> np.ndarray:
    """Return a clip-level log-mel image scaled approximately to ``[0, 1]``."""
    config.validate()
    waveform = _validate_waveform(waveform)
    power = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        power=2.0,
    )
    log_mel = _power_to_log_mel(power)
    return np.clip((log_mel + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)


def extract_log_mel(path: str | Path, config: AudioFeatureConfig) -> np.ndarray:
    return extract_log_mel_from_waveform(load_audio(path, config.sample_rate), config)
