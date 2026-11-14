#!/usr/bin/env python3
"""
Person 3 — Visualization & Validation
Inputs: clean PPG signal, noisy PPG signal, and optional metadata.
Outputs: comparison plots and a validation report.

Supported signal formats: CSV, TXT, DAT, NPY.
CSV files may contain a time column and one signal column. If there are multiple
numeric columns, the script prefers columns whose names contain "ppg", "signal",
"clean", or "noisy"; otherwise it uses the first suitable numeric column.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_signal(path, preferred_words):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 2:
            arr = arr[:, 0]
        values = np.asarray(arr, dtype=float).reshape(-1)
        return values, None

    if suffix in (".txt", ".dat"):
        data = np.genfromtxt(path, comments="#", dtype=float)
        if data.ndim == 0:
            data = np.array([data])
        if data.ndim == 2:
            data = data[:, 0]
        return np.asarray(data, dtype=float).reshape(-1), None

    if suffix == ".csv":
        # Read as named columns first; support comma-separated numeric or headed CSV.
        try:
            table = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding=None)
            if table.dtype.names:
                names = list(table.dtype.names)
                numeric_candidates = []
                for name in names:
                    vals = np.atleast_1d(table[name]).astype(float)
                    if np.count_nonzero(np.isfinite(vals)) > 0:
                        numeric_candidates.append((name, vals))
                if not numeric_candidates:
                    raise ValueError(f"No numeric signal columns found in {path}")
                preferred = [
                    item for item in numeric_candidates
                    if any(word in item[0].lower() for word in preferred_words)
                    and "time" not in item[0].lower()
                    and "timestamp" not in item[0].lower()
                ]
                chosen = preferred[0] if preferred else None
                if chosen is None:
                    non_time = [
                        item for item in numeric_candidates
                        if "time" not in item[0].lower()
                        and "timestamp" not in item[0].lower()
                    ]
                    chosen = non_time[0] if non_time else numeric_candidates[0]
                time_col = next(
                    (vals for name, vals in numeric_candidates
                     if "time" in name.lower() or "timestamp" in name.lower()),
                    None
                )
                return np.asarray(chosen[1], dtype=float).reshape(-1), (
                    np.asarray(time_col, dtype=float).reshape(-1) if time_col is not None else None
                )
        except (ValueError, TypeError, IndexError):
            pass

        # Headerless CSV fallback.
        data = np.genfromtxt(path, delimiter=",", dtype=float, comments="#")
        if data.ndim == 1:
            values = data
            time = None
        elif data.ndim == 2:
            # For two columns, assume col 0 is time and col 1 is signal.
            if data.shape[1] >= 2:
                time, values = data[:, 0], data[:, 1]
            else:
                time, values = None, data[:, 0]
        else:
            raise ValueError(f"Could not interpret CSV file: {path}")
        return np.asarray(values, dtype=float).reshape(-1), (
            np.asarray(time, dtype=float).reshape(-1) if time is not None else None
        )

    raise ValueError(f"Unsupported file format '{suffix}' for {path}. Use CSV, TXT, DAT, or NPY.")


def clean_values(signal):
    signal = np.asarray(signal, dtype=float).reshape(-1)
    finite = np.isfinite(signal)
    if np.count_nonzero(finite) < 3:
        raise ValueError("A signal must contain at least 3 finite samples.")
    if not np.all(finite):
        idx = np.arange(len(signal))
        signal = signal.copy()
        signal[~finite] = np.interp(idx[~finite], idx[finite], signal[finite])
    return signal


def metric_report(clean, noisy, fs):
    lines = []
    n = min(len(clean), len(noisy))
    if len(clean) != len(noisy):
        lines.append(
            f"NOTE: Signal lengths differ (clean={len(clean)}, noisy={len(noisy)}). "
            f"Metrics use the first {n} paired samples."
        )
    c = clean[:n]
    y = noisy[:n]
    error = y - c

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    clean_power = float(np.mean(c ** 2))
    noise_power = float(np.mean(error ** 2))
    if noise_power > 0:
        snr_db = float(10 * np.log10(max(clean_power, 1e-15) / noise_power))
    else:
        snr_db = float("inf")
    corr = float(np.corrcoef(c, y)[0, 1]) if np.std(c) > 0 and np.std(y) > 0 else float("nan")

    lines.extend([
        "",
        "COMPARISON METRICS",
        "------------------",
        f"Paired samples compared: {n}",
        f"Duration of paired segment: {n / fs:.3f} seconds",
        f"Mean absolute error (MAE): {mae:.6f}",
        f"Root mean square error (RMSE): {rmse:.6f}",
        f"Signal-to-error ratio (SER): {snr_db:.3f} dB",
        f"Pearson correlation: {corr:.6f}",
        "",
        "INTERPRETATION",
        "--------------",
        "MAE and RMSE quantify the difference between the supplied clean and noisy signals.",
        "Correlation indicates how similarly the two waveforms vary.",
        "SER is calculated from clean-signal power divided by the power of (noisy - clean).",
        "These metrics are meaningful only if both files refer to the same recording, are aligned,",
        "use compatible units, and are sampled at the same rate. They are not a clinical validation.",
    ])
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Person 3: plot clean/noisy PPG and generate a validation report."
    )
    parser.add_argument("--clean", default="clean_ppg.csv", help="Clean PPG file (default: clean_ppg.csv)")
    parser.add_argument("--noisy", default="noisy_ppg.csv", help="Noisy PPG file (default: noisy_ppg.csv)")
    parser.add_argument("--metadata", default="metadata.json", help="Optional metadata JSON file")
    parser.add_argument("--fs", type=float, default=None, help="Sampling frequency in Hz; overrides metadata")
    parser.add_argument("--outdir", default="hitansh_person3_output", help="Output folder")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metadata = {}
    metadata_path = Path(args.metadata)
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            if not isinstance(metadata, dict):
                metadata = {}
        except Exception as exc:
            print(f"Warning: could not read metadata JSON ({exc}); continuing without it.")

    fs = args.fs or metadata.get("sampling_frequency_hz") or metadata.get("fs") or 125.0
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling frequency must be a positive number.")

    clean, clean_time = load_signal(args.clean, ("clean", "ppg", "signal"))
    noisy, noisy_time = load_signal(args.noisy, ("noisy", "ppg", "signal"))
    clean = clean_values(clean)
    noisy = clean_values(noisy)

    clean_time = clean_time if clean_time is not None and len(clean_time) == len(clean) else np.arange(len(clean)) / fs
    noisy_time = noisy_time if noisy_time is not None and len(noisy_time) == len(noisy) else np.arange(len(noisy)) / fs

    # Individual and overlay plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(clean_time, clean, linewidth=0.8)
    axes[0].set_title("Clean PPG Signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Time (s)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(noisy_time, noisy, linewidth=0.8)
    axes[1].set_title("Noisy PPG Signal")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    n = min(len(clean), len(noisy))
    axes[2].plot(np.arange(n) / fs, clean[:n], label="Clean PPG", linewidth=0.9)
    axes[2].plot(np.arange(n) / fs, noisy[:n], label="Noisy PPG", linewidth=0.8, alpha=0.8)
    axes[2].set_title("Clean vs Noisy PPG (paired samples)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = outdir / "clean_noisy_comparison.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    report = [
        "PERSON 3 — VISUALIZATION & VALIDATION REPORT",
        "============================================",
        f"Clean signal file: {args.clean}",
        f"Noisy signal file: {args.noisy}",
        f"Metadata file: {args.metadata if metadata_path.exists() else 'not supplied'}",
        f"Sampling frequency used: {fs:g} Hz",
        f"Clean signal samples: {len(clean)}",
        f"Noisy signal samples: {len(noisy)}",
        f"Clean signal duration: {len(clean) / fs:.3f} seconds",
        f"Noisy signal duration: {len(noisy) / fs:.3f} seconds",
        f"Comparison plot: {plot_path.name}",
    ]
    if metadata:
        report.extend(["", "METADATA", "--------", json.dumps(metadata, indent=2, ensure_ascii=False)])
    report.extend(metric_report(clean, noisy, fs))
    report_path = outdir / "validation_report.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("Person 3 visualization and validation completed.")
    print(f"Plot saved:   {plot_path}")
    print(f"Report saved: {report_path}")
    print(f"Sampling frequency: {fs:g} Hz")
    print(f"Clean samples: {len(clean)} | Noisy samples: {len(noisy)}")
    print("Reminder: verify both signals are aligned and sampled at the same rate before interpreting metrics.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Check that the clean/noisy input filenames and formats are correct.", file=sys.stderr)
        sys.exit(1)
