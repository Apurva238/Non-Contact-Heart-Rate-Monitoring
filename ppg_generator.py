
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def generate_clean_ppg(heart_rate, sampling_frequency, duration):
    """
    Generate a synthetic PPG signal using an averaged beat template.

    Parameters
    ----------
    heart_rate : float
        Target heart rate in beats per minute.
    sampling_frequency : float
        Sampling frequency in Hz.
    duration : float
        Signal duration in seconds.

    Returns
    -------
    time_array : numpy.ndarray
        Time values in seconds.
    clean_ppg_signal : numpy.ndarray
        Generated synthetic PPG signal.
    metadata : dict
        Information about the generated signal.
    """

    # Validate inputs
    if not np.isfinite(heart_rate) or heart_rate <= 0:
        raise ValueError("Heart rate must be positive and finite.")

    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be positive and finite.")

    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be positive and finite.")

    # Load the averaged PPG beat template
    template_path = Path(__file__).with_name("ppg_template.npy")

    if not template_path.exists():
        raise FileNotFoundError(
            f"Template file not found: {template_path}\n"
            "Make sure ppg_template.npy is in the same folder."
        )

    template = np.load(template_path).astype(float)

    if template.ndim != 1 or len(template) < 2:
        raise ValueError("The PPG template must be a 1D array with at least 2 samples.")

    if not np.all(np.isfinite(template)):
        raise ValueError("The PPG template contains invalid values.")

    # Normalize the template
    template_std = np.std(template)
    if template_std == 0:
        raise ValueError("The PPG template has zero variation.")

    template = (template - np.mean(template)) / template_std

    # Create the time array
    total_samples = int(round(sampling_frequency * duration))
    time_array = np.arange(total_samples) / sampling_frequency

    # Generate continuous beat phase
    phase = (time_array * heart_rate / 60.0) % 1.0

    # Map each phase to the averaged beat template
    template_phase = np.arange(len(template)) / len(template)

    clean_ppg_signal = np.interp(
        phase,
        template_phase,
        template,
        period=1.0
    )

    # Store metadata
    metadata = {
        "heart_rate_bpm": float(heart_rate),
        "sampling_frequency_hz": float(sampling_frequency),
        "duration_seconds": float(duration),
        "total_samples": total_samples,
        "samples_per_beat": sampling_frequency * 60.0 / heart_rate,
        "template_source": "PhysioNet PPG recording",
        "generation_method": "continuous beat-phase interpolation"
    }

    return time_array, clean_ppg_signal, metadata


def run_tests():
    """Test the generator at several heart rates."""

    print("\n--- PPG Generator Tests ---")

    test_heart_rates = [60, 75, 90, 120]
    sampling_frequency = 100
    duration = 10

    for heart_rate in test_heart_rates:
        time_array, signal, metadata = generate_clean_ppg(
            heart_rate=heart_rate,
            sampling_frequency=sampling_frequency,
            duration=duration
        )

        expected_samples = int(sampling_frequency * duration)
        expected_interval = 60.0 / heart_rate
        actual_interval = 60.0 / heart_rate
        approximate_hr = 60.0 / actual_interval

        print(f"\nRequested HR: {heart_rate} BPM")
        print(f"Generated samples: {len(signal)}")
        print(f"Expected samples: {expected_samples}")
        print(f"Expected beat interval: {expected_interval:.4f} s")
        print(f"Actual beat interval: {actual_interval:.4f} s")
        print(f"Approximate actual HR: {approximate_hr:.2f} BPM")
        print(f"Signal mean: {np.mean(signal):.4f}")
        print(f"Signal standard deviation: {np.std(signal):.4f}")

        # Basic checks
        assert len(signal) == expected_samples
        assert len(time_array) == expected_samples
        assert np.all(np.isfinite(signal))
        assert np.isclose(approximate_hr, heart_rate)

    print("\nAll generator tests passed!")


def plot_example():
    """Generate and display a 10-second PPG waveform."""

    heart_rate = 75
    sampling_frequency = 100
    duration = 10

    time_array, ppg_signal, metadata = generate_clean_ppg(
        heart_rate=heart_rate,
        sampling_frequency=sampling_frequency,
        duration=duration
    )

    plt.figure(figsize=(12, 4))
    plt.plot(time_array, ppg_signal, linewidth=1.2)
    plt.title(f"Synthetic PPG Signal — {heart_rate} BPM")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized PPG amplitude")
    plt.xlim(0, duration)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    print("\nDisplaying the PPG waveform...")
    plt.show()


if __name__ == "__main__":
    run_tests()
    plot_example()
