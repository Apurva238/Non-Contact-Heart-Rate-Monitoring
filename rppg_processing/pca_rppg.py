"""Frequency-informed PCA extraction for camera-based rPPG (Part 2).

Input CSV contract (from Part 1): timestamp_s, red, green, blue; optional valid.
Output: processed_rppg.csv with timestamp_s, rppg, valid.

This is a research/demo heuristic, not a medical device or a guarantee that the
selected periodic component is physiological. Motion can also be periodic.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt, welch


@dataclass
class ComponentScore:
    component: str
    selected: bool
    score: float
    peak_frequency_hz: float
    estimated_bpm: float
    peak_prominence_ratio: float
    autocorrelation_at_period: float
    explained_variance_ratio: float


def _estimate_fs(timestamps: np.ndarray) -> float:
    diffs = np.diff(timestamps)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size < 2:
        raise ValueError("Not enough increasing timestamps to estimate sampling rate.")
    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        raise ValueError("Timestamp differences must be positive.")
    return 1.0 / median_dt


def _uniform_resample(
    timestamps: np.ndarray, rgb: np.ndarray, valid: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate valid RGB rows onto a uniform time grid at the estimated rate."""
    finite = np.isfinite(timestamps) & np.all(np.isfinite(rgb), axis=1) & valid
    if finite.sum() < 8:
        raise ValueError("Fewer than 8 valid RGB samples remain after validation.")
    t = timestamps[finite]
    x = rgb[finite]
    order = np.argsort(t)
    t, x = t[order], x[order]
    # Duplicate timestamps make interpolation ambiguous; average their RGB values.
    unique_t, inverse = np.unique(t, return_inverse=True)
    if unique_t.size != t.size:
        sums = np.zeros((unique_t.size, 3), dtype=float)
        counts = np.zeros(unique_t.size, dtype=float)
        np.add.at(sums, inverse, x)
        np.add.at(counts, inverse, 1.0)
        x = sums / counts[:, None]
        t = unique_t
    if t.size < 8 or t[-1] <= t[0]:
        raise ValueError("Insufficient time span after removing duplicate timestamps.")
    step = 1.0 / fs
    grid = np.arange(t[0], t[-1] + step * 0.25, step)
    if grid.size < 8:
        raise ValueError("Recording is too short; at least 8 resampled points are needed.")
    interpolated = np.column_stack([np.interp(grid, t, x[:, ch]) for ch in range(3)])
    return grid, interpolated, finite


def _bandpass(signal: np.ndarray, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = fs / 2.0
    if not (0 < low_hz < high_hz < nyquist):
        raise ValueError(
            f"Bandpass must satisfy 0 < low < high < Nyquist ({nyquist:.3f} Hz); "
            f"got {low_hz}–{high_hz} Hz."
        )
    sos = butter(3, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    if len(signal) <= padlen:
        raise ValueError(
            f"Recording is too short for zero-phase filtering ({len(signal)} samples; "
            f"need more than {padlen}). Use a longer video clip."
        )
    return sosfiltfilt(sos, signal)


def _score_component(
    signal: np.ndarray, fs: float, low_hz: float, high_hz: float, explained: float
) -> tuple[np.ndarray, dict[str, float]]:
    filtered = _bandpass(signal, fs, low_hz, high_hz)
    filtered = filtered - np.mean(filtered)
    f, pxx = welch(filtered, fs=fs, nperseg=min(len(filtered), max(32, int(fs * 8))))
    band = (f >= low_hz) & (f <= high_hz)
    if band.sum() < 3:
        raise ValueError("Too few spectral bins in the heart-rate band; use a longer clip.")
    fb, pb = f[band], pxx[band]
    peak_idx = int(np.argmax(pb))
    peak_freq = float(fb[peak_idx])
    background = float(np.median(pb) + np.finfo(float).eps)
    peak_ratio = float(pb[peak_idx] / background)
    # Normalized autocorrelation at the lag implied by the dominant spectral peak.
    lag = int(round(fs / peak_freq)) if peak_freq > 0 else 0
    centered = filtered - np.mean(filtered)
    denom = float(np.dot(centered, centered)) + np.finfo(float).eps
    ac = float(np.dot(centered[:-lag], centered[lag:]) / denom) if 1 <= lag < len(centered) else 0.0
    # Fixed, transparent heuristic: log peak prominence plus periodicity support.
    score = float(math.log(max(peak_ratio, 1.0)) + max(0.0, ac))
    details = {
        "score": score,
        "peak_frequency_hz": peak_freq,
        "estimated_bpm": peak_freq * 60.0,
        "peak_prominence_ratio": peak_ratio,
        "autocorrelation_at_period": ac,
        "explained_variance_ratio": float(explained),
    }
    return filtered, details


def process_rgb_csv(
    input_csv: str | Path,
    output_dir: str | Path,
    low_bpm: float = 42.0,
    high_bpm: float = 180.0,
    sampling_rate_hz: float | None = None,
) -> dict[str, Any]:
    """Run validation, RGB PCA, frequency-informed component selection and filtering."""
    input_path = Path(input_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    required = {"timestamp_s", "red", "green", "blue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(sorted(missing))}")

    timestamps = pd.to_numeric(df["timestamp_s"], errors="coerce").to_numpy(dtype=float)
    rgb = df[["red", "green", "blue"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if "valid" in df.columns:
        valid_col = df["valid"].astype(str).str.strip().str.lower()
        valid = valid_col.isin(["1", "true", "yes", "y"]).to_numpy()
    else:
        valid = np.ones(len(df), dtype=bool)
    valid = valid & np.isfinite(timestamps) & np.all(np.isfinite(rgb), axis=1)
    if valid.sum() < 8:
        raise ValueError("Input contains fewer than 8 valid RGB rows.")

    # Sort input by timestamp before estimating rate; invalid/non-finite rows are ignored.
    usable_t = timestamps[valid]
    order = np.argsort(usable_t)
    sorted_t = usable_t[order]
    fs_est = _estimate_fs(sorted_t)
    fs = float(sampling_rate_hz) if sampling_rate_hz is not None else fs_est
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling rate must be a positive finite number.")
    low_hz, high_hz = low_bpm / 60.0, high_bpm / 60.0
    if low_bpm <= 0 or high_bpm <= low_bpm:
        raise ValueError("BPM limits must satisfy 0 < low_bpm < high_bpm.")
    if high_hz >= fs / 2.0:
        raise ValueError(
            f"High cutoff {high_hz:.3f} Hz must be below Nyquist {fs / 2.0:.3f} Hz. "
            "Increase sampling rate or lower --high-bpm."
        )

    grid, rgb_uniform, _ = _uniform_resample(timestamps, rgb, valid, fs)
    # Remove slow trends before PCA and standardize channels so scale does not dominate.
    cleaned = np.column_stack([detrend(rgb_uniform[:, i], type="linear") for i in range(3)])
    scales = np.std(cleaned, axis=0, ddof=1)
    scales[scales < 1e-12] = 1.0
    standardized = (cleaned - np.mean(cleaned, axis=0)) / scales
    # SVD PCA; component signs are arbitrary, which is fine for frequency scoring.
    _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    scores = standardized @ vt.T
    variance = singular_values**2
    explained = variance / variance.sum() if variance.sum() > 0 else np.zeros(3)

    filtered_components: list[np.ndarray] = []
    component_details: list[dict[str, Any]] = []
    for i in range(3):
        filtered, details = _score_component(scores[:, i], fs, low_hz, high_hz, float(explained[i]))
        filtered_components.append(filtered)
        component_details.append({"component": f"PC{i + 1}", **details})
    best_index = int(np.argmax([row["score"] for row in component_details]))
    selected = filtered_components[best_index]
    selected = (selected - np.mean(selected)) / (np.std(selected) + np.finfo(float).eps)

    output = pd.DataFrame({"timestamp_s": grid, "rppg": selected, "valid": True})
    output.to_csv(out_dir / "processed_rppg.csv", index=False)
    for row in component_details:
        row["selected"] = row["component"] == f"PC{best_index + 1}"
    scores_df = pd.DataFrame(component_details)
    scores_df.to_csv(out_dir / "component_scores.csv", index=False)

    metadata: dict[str, Any] = {
        "method": "frequency-informed PCA component selection",
        "input_file": str(input_path),
        "output_file": "processed_rppg.csv",
        "sampling_rate_hz_used": fs,
        "sampling_rate_hz_estimated_from_timestamps": fs_est,
        "sampling_rate_was_overridden": sampling_rate_hz is not None,
        "low_bpm": float(low_bpm),
        "high_bpm": float(high_bpm),
        "low_cutoff_hz": float(low_hz),
        "high_cutoff_hz": float(high_hz),
        "pca_component_selected": f"PC{best_index + 1}",
        "selection_score": "log(peak PSD / median in-band PSD) + max(0, autocorrelation at peak period)",
        "component_scores": component_details,
        "input_rows": int(len(df)),
        "valid_input_rows": int(valid.sum()),
        "output_rows": int(len(grid)),
        "duration_s": float(grid[-1] - grid[0]),
        "caveat": "Exploratory signal-processing output; periodic motion or lighting can be mistaken for pulse. Validate against reference heart rate.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract an rPPG signal from Part 1 RGB CSV data.")
    parser.add_argument("input_csv", help="CSV with timestamp_s, red, green, blue, optional valid")
    parser.add_argument("--output-dir", default="rppg_output", help="Directory for output CSV and metadata")
    parser.add_argument("--low-bpm", type=float, default=42.0, help="Lower plausible heart-rate bound (default: 42)")
    parser.add_argument("--high-bpm", type=float, default=180.0, help="Upper plausible heart-rate bound (default: 180)")
    parser.add_argument("--fs", type=float, default=None, help="Override sampling rate in Hz; default estimates from timestamps")
    args = parser.parse_args()
    metadata = process_rgb_csv(args.input_csv, args.output_dir, args.low_bpm, args.high_bpm, args.fs)
    print(f"Selected component: {metadata['pca_component_selected']}")
    print(f"Sampling rate: {metadata['sampling_rate_hz_used']:.3f} Hz")
    print(f"Wrote: {Path(args.output_dir) / 'processed_rppg.csv'}")
    print(f"Wrote: {Path(args.output_dir) / 'component_scores.csv'}")
    print(f"Wrote: {Path(args.output_dir) / 'metadata.json'}")


if __name__ == "__main__":
    main()
