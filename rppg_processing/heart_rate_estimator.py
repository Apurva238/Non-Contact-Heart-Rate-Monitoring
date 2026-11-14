import numpy as np
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt, welch


def estimate_heart_rate(signal, sampling_rate, min_bpm=45, max_bpm=180):
    """Estimate BPM using Welch PSD and peak intervals."""
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be a positive number.")
    if not (0 < min_bpm < max_bpm):
        raise ValueError("BPM limits are invalid.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or infinite values.")
    if len(signal) < int(np.ceil(sampling_rate * 5)):
        raise ValueError("At least 5 seconds of signal is required.")
    if np.std(signal) < 1e-10:
        raise ValueError("Signal is effectively constant.")

    low_hz, high_hz = min_bpm / 60.0, max_bpm / 60.0
    if high_hz >= sampling_rate / 2:
        raise ValueError("Sampling rate is too low for the selected maximum BPM.")

    signal = detrend(signal)
    sos = butter(3, [low_hz, high_hz], btype="bandpass",
                 fs=sampling_rate, output="sos")
    try:
        filtered = sosfiltfilt(sos, signal)
    except ValueError as exc:
        raise ValueError("Signal is too short for reliable filtering.") from exc

    if not np.all(np.isfinite(filtered)) or np.std(filtered) < 1e-10:
        raise ValueError("Filtered signal is invalid or has insufficient variation.")

    nperseg = min(len(filtered), max(1, int(sampling_rate * 30)))
    frequencies, power = welch(filtered, fs=sampling_rate, nperseg=nperseg)
    valid_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(valid_mask) or not np.any(power[valid_mask] > 0):
        raise ValueError("No usable pulse-band power was found.")

    valid_indices = np.flatnonzero(valid_mask)
    ranked_indices = valid_indices[
        np.argsort(power[valid_mask])[::-1]
    ]

    print("\nTop 5 spectral peaks:")
    for idx in ranked_indices[:5]:
        print(
            f"  {frequencies[idx] * 60:.2f} BPM "
            f"(power: {power[idx]:.6f})"
        )

    best_index = int(ranked_indices[0])

    # Refine the spectral peak using parabolic interpolation.
    # This estimates the peak between neighboring frequency bins.
    delta = 0.0

    if 0 < best_index < len(power) - 1:
        left = float(power[best_index - 1])
        center = float(power[best_index])
        right = float(power[best_index + 1])

        denominator = left - 2.0 * center + right

        if denominator != 0 and np.isfinite(denominator):
            delta = 0.5 * (left - right) / denominator
            delta = float(np.clip(delta, -1.0, 1.0))

    frequency_hz = (
        frequencies[best_index]
        + delta * (frequencies[1] - frequencies[0])
    )

    frequency_bpm = float(frequency_hz * 60.0)


    min_distance = max(1, int(sampling_rate * 60.0 / max_bpm))
    prominence = max(float(np.std(filtered) * 0.2), 1e-12)
    peaks, _ = find_peaks(filtered, distance=min_distance, prominence=prominence)

    peak_bpm = None
    valid_intervals_count = 0
    if len(peaks) >= 2:
        intervals = np.diff(peaks) / sampling_rate
        valid_intervals = intervals[
            (intervals >= 60.0 / max_bpm) & (intervals <= 60.0 / min_bpm)
        ]
        valid_intervals_count = int(len(valid_intervals))
        if valid_intervals_count:
            peak_bpm = float(60.0 / np.median(valid_intervals))

    difference = abs(frequency_bpm - peak_bpm) if peak_bpm is not None else None
    methods_agree = bool(peak_bpm is not None and difference <= 5.0)

    total_power = float(np.sum(power))
    concentration = (
        float(np.sum(power[valid_mask]) / total_power) if total_power > 0 else 0.0
    )

    return {
        "heart_rate_bpm": round(frequency_bpm, 2),
        "frequency_estimate_bpm": round(frequency_bpm, 2),
        "peak_estimate_bpm": round(peak_bpm, 2) if peak_bpm is not None else None,
        "method_difference_bpm": round(difference, 2) if difference is not None else None,
        "methods_agree": methods_agree,
        "consistency_check_passed": methods_agree,
        "signal_duration_seconds": round(len(signal) / sampling_rate, 2),
        "sampling_rate_hz": float(sampling_rate),
        "detected_peaks": int(len(peaks)),
        "valid_peak_intervals": valid_intervals_count,
        "spectral_concentration": round(concentration, 4),
        "quality_check_passed": methods_agree,
        "quality_note": (
            "Frequency and peak estimates agree; this does not guarantee accuracy."
            if methods_agree else
            "Review estimate: methods disagree or valid peak intervals are insufficient."
        ),
    }


if __name__ == "__main__":
    fs, duration, expected_bpm = 100, 30, 75
    t = np.arange(0, duration, 1 / fs)
    rng = np.random.default_rng(42)
    f = expected_bpm / 60.0
    x = (np.sin(2*np.pi*f*t) + 0.15*np.sin(2*np.pi*2*f*t)
         + rng.normal(0, 0.1, len(t)))
    result = estimate_heart_rate(x, fs)
    print("Synthetic heart-rate demonstration")
    print("Expected BPM:", expected_bpm)
    print("Frequency estimate:", result["frequency_estimate_bpm"])
    print("Peak estimate:", result["peak_estimate_bpm"])
    print("Diagnostic:", result["quality_note"])
