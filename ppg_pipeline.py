from ppg_generator import generate_clean_ppg
from noise_simulation import add_disturbances


def generate_ppg_with_noise(
    heart_rate,
    sampling_frequency,
    duration,
    noise_level=0.01,
    drift_amplitude=0.03,
    drift_frequency=0.1,
    amplitude_variation=0.1,
    amplitude_frequency=0.05,
    seed=42,
):
    # Step 1: Generate clean PPG
    time_array, clean_signal, metadata = generate_clean_ppg(
        heart_rate=heart_rate,
        sampling_frequency=sampling_frequency,
        duration=duration,
    )

    # Step 2: Add simulated disturbances
    signals = add_disturbances(
        clean_signal=clean_signal,
        sampling_rate=sampling_frequency,
        noise_level=noise_level,
        drift_amplitude=drift_amplitude,
        drift_frequency=drift_frequency,
        amplitude_variation=amplitude_variation,
        amplitude_frequency=amplitude_frequency,
        seed=seed,
    )

    # Step 3: Return all signals and metadata
    return time_array, signals, metadata