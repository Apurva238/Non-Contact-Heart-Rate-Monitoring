"""
noise_simulation.py

Pranjal's LSSP project module:
Adds simulated baseline drift, Gaussian noise, and slow pulse-amplitude
variation to a clean PPG signal.

Input:
    clean_signal: 1D array of PPG samples from Apurva's model
    sampling_rate: sampling frequency in Hz

Output:
    Dictionary containing clean, individually disturbed, and combined signals.

Note:
    Disturbance magnitudes are expressed in the same amplitude units as the
    input signal. Calibrate them after inspecting Apurva's model output.
"""

import numpy as np


def _validate_signal_and_sampling_rate(clean_signal, sampling_rate):
    """Convert the input to a 1D float array and validate core inputs."""
    signal = np.asarray(clean_signal, dtype=float)

    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("clean_signal must be a non-empty 1D array.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("clean_signal must contain only finite values.")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be a positive finite number.")

    return signal


def _validate_nonnegative_parameters(**parameters):
    for name, value in parameters.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite, non-negative number.")


def add_baseline_drift(
    clean_signal,
    sampling_rate,
    drift_amplitude=0.03,
    drift_frequency=0.1,
):
    """Return a copy of the clean signal with sinusoidal baseline drift."""
    signal = _validate_signal_and_sampling_rate(clean_signal, sampling_rate)
    _validate_nonnegative_parameters(
        drift_amplitude=drift_amplitude,
        drift_frequency=drift_frequency,
    )

    t = np.arange(signal.size) / sampling_rate
    drift = drift_amplitude * np.sin(2 * np.pi * drift_frequency * t)
    return signal + drift


def add_gaussian_noise(clean_signal, noise_level=0.01, seed=42):
    """
    Return a copy of the clean signal with additive Gaussian noise.

    noise_level is the standard deviation of the noise, in signal-amplitude
    units. A fixed seed makes experiments reproducible.
    """
    signal = np.asarray(clean_signal, dtype=float)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("clean_signal must be a non-empty 1D array.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("clean_signal must contain only finite values.")
    _validate_nonnegative_parameters(noise_level=noise_level)

    rng = np.random.default_rng(seed)
    noise = noise_level * rng.standard_normal(signal.size)
    return signal + noise


def add_amplitude_variation(
    clean_signal,
    sampling_rate,
    variation=0.1,
    modulation_frequency=0.05,
):
    """
    Return a copy of the clean signal with slow multiplicative amplitude
    modulation. variation=0.1 means the multiplier ranges from 0.9 to 1.1.
    """
    signal = _validate_signal_and_sampling_rate(clean_signal, sampling_rate)
    _validate_nonnegative_parameters(
        variation=variation,
        modulation_frequency=modulation_frequency,
    )
    if variation > 1:
        raise ValueError("variation must be between 0 and 1.")

    t = np.arange(signal.size) / sampling_rate
    envelope = 1 + variation * np.sin(
        2 * np.pi * modulation_frequency * t
    )
    return envelope * signal


def add_disturbances(
    clean_signal,
    sampling_rate,
    noise_level=0.01,
    drift_amplitude=0.03,
    drift_frequency=0.1,
    amplitude_variation=0.1,
    amplitude_frequency=0.05,
    seed=42,
):
    """
    Generate separate disturbance experiments and a combined signal.

    The combined signal is:
        amplitude_envelope * clean_signal + baseline_drift + Gaussian_noise

    Returns a dictionary with keys:
        clean, drift_only, noise_only, amplitude_only, combined
    """
    signal = _validate_signal_and_sampling_rate(clean_signal, sampling_rate)
    _validate_nonnegative_parameters(
        noise_level=noise_level,
        drift_amplitude=drift_amplitude,
        drift_frequency=drift_frequency,
        amplitude_variation=amplitude_variation,
        amplitude_frequency=amplitude_frequency,
    )
    if amplitude_variation > 1:
        raise ValueError("amplitude_variation must be between 0 and 1.")

    drift_signal = add_baseline_drift(
        signal, sampling_rate, drift_amplitude, drift_frequency
    )
    noise_signal = add_gaussian_noise(signal, noise_level, seed)
    amplitude_signal = add_amplitude_variation(
        signal, sampling_rate, amplitude_variation, amplitude_frequency
    )

    t = np.arange(signal.size) / sampling_rate
    drift = drift_amplitude * np.sin(
        2 * np.pi * drift_frequency * t
    )
    rng = np.random.default_rng(seed)
    noise = noise_level * rng.standard_normal(signal.size)
    envelope = 1 + amplitude_variation * np.sin(
        2 * np.pi * amplitude_frequency * t
    )
    combined_signal = envelope * signal + drift + noise

    return {
        "clean": signal.copy(),
        "drift_only": drift_signal,
        "noise_only": noise_signal,
        "amplitude_only": amplitude_signal,
        "combined": combined_signal,
    }
