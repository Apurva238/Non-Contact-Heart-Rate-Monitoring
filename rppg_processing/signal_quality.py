import numpy as np
from scipy.signal import welch


def assess_signal_quality(signal, sampling_rate):
    """Basic rPPG signal checks; not a clinical accuracy guarantee."""
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")
    if len(signal) < 5:
        return {"quality": "poor", "usable": False, "reason": "Too few samples."}
    if not np.all(np.isfinite(signal)):
        return {"quality": "poor", "usable": False,
                "reason": "Signal contains NaN or infinite values."}

    duration = len(signal) / sampling_rate
    std = float(np.std(signal))
    if duration < 5:
        return {"quality": "poor", "usable": False,
                "reason": "At least 5 seconds of signal is required."}
    if std < 1e-10:
        return {"quality": "poor", "usable": False,
                "reason": "Signal has almost no variation."}

    frequencies, power = welch(
        signal, fs=sampling_rate,
        nperseg=min(len(signal), max(1, int(sampling_rate * 10)))
    )
    heart_band = (frequencies >= 0.75) & (frequencies <= 3.0)
    if not np.any(heart_band):
        return {"quality": "poor", "usable": False,
                "reason": "Insufficient frequency resolution."}

    total_power = float(np.sum(power))
    if total_power <= 0:
        return {"quality": "poor", "usable": False, "reason": "Signal power is zero."}

    ratio = float(np.sum(power[heart_band]) / total_power)
    quality = "good" if ratio >= 0.5 else "moderate" if ratio >= 0.25 else "poor"
    return {
        "quality": quality,
        "usable": quality != "poor",
        "duration_seconds": round(duration, 2),
        "standard_deviation": round(std, 6),
        "heart_band_power_ratio": round(ratio, 4),
        "reason": "Basic signal checks completed; this is not a clinical accuracy guarantee."
    }
