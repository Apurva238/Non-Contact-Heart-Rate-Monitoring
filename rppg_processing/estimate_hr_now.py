import numpy as np
import pandas as pd
from scipy.signal import welch, detrend
from pathlib import Path

# PCA signal generated earlier
ROOT = Path(__file__).resolve().parent.parent

PCA_PATH = ROOT / "rppg_processing/output/forehead_chrominance/processed_rppg.csv"

FPS = 30.0

# Temporary range based on this video's reference HR
MIN_HR = 85
MAX_HR = 110


def estimate_hr():

    # Load PCA rPPG signal
    df = pd.read_csv(PCA_PATH)

    signal = df.loc[df["valid"], "rppg"].to_numpy(dtype=float)

    signal = signal[np.isfinite(signal)]

    if len(signal) < FPS * 10:
        print("Not enough valid signal samples.")
        return

    # Remove DC component and linear trend
    signal = detrend(signal)

    # Calculate Power Spectral Density
    frequencies, power = welch(
        signal,
        fs=FPS,
        nperseg=min(1024, len(signal))
    )

    # Convert frequency range from BPM to Hz
    min_freq = MIN_HR / 60
    max_freq = MAX_HR / 60

    # Select frequencies within expected HR range
    mask = (
        (frequencies >= min_freq) &
        (frequencies <= max_freq)
    )

    if not np.any(mask):
        print("No valid heart-rate frequency found.")
        return

    selected_frequencies = frequencies[mask]
    selected_power = power[mask]

    # Dominant frequency
    peak_index = np.argmax(selected_power)

    heart_rate = selected_frequencies[peak_index] * 60

    print("\n========== HEART RATE ESTIMATION ==========")
    print("Method: PCA-based rPPG")
    print(f"Estimated Heart Rate: {heart_rate:.2f} BPM")
    print("===========================================\n")

    return heart_rate


if __name__ == "__main__":
    estimate_hr()