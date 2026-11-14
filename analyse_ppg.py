import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.signal import find_peaks

# Load dataset
df = pd.read_csv("s1_sit.csv")

# Sampling rate
fs = 500

# Analyze 5 seconds from the first recording segment
start = 1000
end = start + 5 * fs

# Design bandpass filter
sos = butter(
    4,
    [0.5, 5.0],
    btype="bandpass",
    fs=fs,
    output="sos"
)

# Plot filtered signals
fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)

for i in range(1, 7):
    column = f"pleth_{i}"
    signal = df[column].iloc[start:end].to_numpy(dtype=float)

    # Remove the mean and filter
    signal = signal - np.mean(signal)
    filtered_signal = sosfiltfilt(sos, signal)

    if i in [1, 2]:
        peaks, _ = find_peaks(
        filtered_signal,
        distance=int(0.4 * fs),
        prominence=20
    )

    intervals = np.diff(peaks) / fs
    heart_rate = 60 / np.median(intervals)

    print(f"{column}: {len(peaks)} peaks, "
          f"estimated HR = {heart_rate:.1f} BPM")

    time = np.arange(len(signal)) / fs

    axes[i - 1].plot(time, filtered_signal)
    axes[i - 1].set_ylabel(column)
    axes[i - 1].grid(True)

axes[-1].set_xlabel("Time (seconds)")
fig.suptitle("Filtered PPG Channels — 5-Second Segment")
plt.tight_layout()
plt.show()

# Step 1: Extract individual PPG beats

from scipy.signal import find_peaks

# Use a longer segment from the first recording section
start = 1000
end = 11000  # 20 seconds at 500 Hz

signal = df["pleth_2"].iloc[start:end].to_numpy(dtype=float)

# Remove baseline variations
signal = signal - np.mean(signal)
filtered_signal = sosfiltfilt(sos, signal)

# Detect pulse peaks
peaks, _ = find_peaks(
    filtered_signal,
    distance=int(0.4 * fs),
    prominence=20
)

print("\nNumber of detected peaks:", len(peaks))

# Detect and evaluate PPG peaks

start = 1000
end = 11000  # 20 seconds at 500 Hz

signal = df["pleth_2"].iloc[start:end].to_numpy(dtype=float)
signal = signal - np.mean(signal)
filtered_signal = sosfiltfilt(sos, signal)

peaks, _ = find_peaks(
    filtered_signal,
    distance=int(0.4 * fs),
    prominence=20
)

intervals = np.diff(peaks) / fs
heart_rate = 60 / np.median(intervals) if len(intervals) else 0

print("\n--- PPG Peak Analysis ---")
print("Detected peaks:", len(peaks))
print("Estimated heart rate:", round(heart_rate, 2), "BPM")
print("Median beat interval:",
      round(np.median(intervals), 4) if len(intervals) else "N/A",
      "seconds")
print("Beat interval variability (SD):",
      round(np.std(intervals), 4) if len(intervals) else "N/A",
      "seconds")


# Build a template from multiple consistent heartbeat cycles
beat_length = 500
beats = []

if len(peaks) < 4:
    raise ValueError("Not enough detected peaks to build a template.")

# Keep cycles whose durations are close to the median duration
intervals = np.diff(peaks)
median_interval = np.median(intervals)

for j in range(len(peaks) - 1):
    start = peaks[j]
    end = peaks[j + 1]
    interval_length = end - start

    # Reject unusually short or long cycles
    if abs(interval_length - median_interval) > 0.20 * median_interval:
        continue

    beat = filtered_signal[start:end]

    # Resample each cycle to the same number of samples
    beat_resampled = np.interp(
        np.linspace(0, len(beat) - 1, beat_length),
        np.arange(len(beat)),
        beat
    )
    beats.append(beat_resampled)

if len(beats) < 3:
    raise ValueError("Too few consistent beats remained after filtering.")

clean_template = np.mean(beats, axis=0)

# Normalize the averaged template
std = np.std(clean_template)
if std == 0:
    raise ValueError("Template has zero variance.")

clean_template = (
    clean_template - np.mean(clean_template)
) / std

np.save("ppg_template.npy", clean_template)

print("\n--- Improved Template ---")
print("Consistent beats averaged:", len(beats))
print("Template samples:", len(clean_template))
print("Mean:", round(float(np.mean(clean_template)), 4))
print("Standard deviation:", round(float(np.std(clean_template)), 4))
print(
    "Beat boundary jump:",
    round(float(clean_template[0] - clean_template[-1]), 4)
)
print(
    "Largest adjacent-sample change:",
    round(float(np.max(np.abs(np.diff(clean_template)))), 4)
)

    
import numpy as np
import matplotlib.pyplot as plt

# Load the saved heartbeat template
template = np.load("ppg_template.npy")

# Basic numerical checks
print("\n--- Template Validation ---")
print("Number of samples:", len(template))
print("Minimum value:", round(float(np.min(template)), 4))
print("Maximum value:", round(float(np.max(template)), 4))
print("Mean:", round(float(np.mean(template)), 4))
print("Standard deviation:", round(float(np.std(template)), 4))

# Check the jump when one beat joins the next
boundary_jump = template[0] - template[-1]
largest_internal_jump = np.max(np.abs(np.diff(template)))

print("Beat boundary jump:", round(float(boundary_jump), 4))
print("Largest adjacent-sample change:", round(float(largest_internal_jump), 4))

# Display two consecutive copies of the template
repeated_template = np.tile(template, 2)
time = np.arange(len(repeated_template)) / len(template)

plt.figure(figsize=(10, 4))
plt.plot(time, repeated_template)
plt.axvline(1, linestyle="--", label="Beat boundary")
plt.title("Two Repeated PPG Template Cycles")
plt.xlabel("Time (normalized beat cycles)")
plt.ylabel("Normalized PPG amplitude")
plt.legend()
plt.tight_layout()
plt.show()


# --- Multi-channel PPG quality assessment ---
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks

df = pd.read_csv("s1_sit.csv")

fs = 500
start = 1000
end = start + 20 * fs

# Use the same time segment for all channels
sos = butter(4, [0.5, 5.0], btype="bandpass",
             fs=fs, output="sos")

channel_results = []

for channel in [f"pleth_{i}" for i in range(1, 7)]:
    raw = df[channel].iloc[start:end].to_numpy(dtype=float)

    if not np.all(np.isfinite(raw)):
        print(f"{channel}: skipped (non-finite values)")
        continue

    filtered = sosfiltfilt(sos, raw - np.mean(raw))

    # Robust, consistent peak-detection settings for comparison
    peaks, _ = find_peaks(
        filtered,
        distance=int(0.4 * fs),
        prominence=0.5 * np.std(filtered)
    )

    if len(peaks) >= 3:
        intervals = np.diff(peaks) / fs
        estimated_hr = 60 / np.median(intervals)
        interval_cv = (
            np.std(intervals) / np.mean(intervals)
            if np.mean(intervals) > 0 else np.nan
        )
    else:
        estimated_hr = np.nan
        interval_cv = np.nan

    channel_results.append({
        "Channel": channel,
        "Filtered std": np.std(filtered),
        "Detected peaks": len(peaks),
        "Estimated HR (BPM)": estimated_hr,
        "Interval CV": interval_cv
    })

results = pd.DataFrame(channel_results)

print("\n--- PPG Channel Comparison ---")
print(results.round(3).to_string(index=False))


# --- Exploratory PCA fusion of selected PPG channels ---
from sklearn.decomposition import PCA

selected_channels = ["pleth_1", "pleth_2", "pleth_5", "pleth_6"]

filtered_signals = []

for channel in selected_channels:
    raw = df[channel].iloc[start:end].to_numpy(dtype=float)
    filtered = sosfiltfilt(sos, raw - np.mean(raw))
    filtered_signals.append(filtered)

# Each column represents one PPG channel.
X = np.column_stack(filtered_signals)

# Standardize each channel before PCA.
channel_means = np.mean(X, axis=0)
channel_stds = np.std(X, axis=0)

if np.any(channel_stds == 0):
    raise ValueError("A selected channel has zero variance.")

X_standardized = (X - channel_means) / channel_stds

pca = PCA()
pca_signals = pca.fit_transform(X_standardized)

# First principal component combines shared variation across channels.
pc1 = pca_signals[:, 0]

# Peak-based estimate for PC1
pc1_peaks, _ = find_peaks(
    pc1,
    distance=int(0.4 * fs),
    prominence=0.5 * np.std(pc1)
)

print("\n--- PCA Fusion Results ---")
print("Channels used:", selected_channels)
print(
    "Variance explained by PC1:",
    round(float(pca.explained_variance_ratio_[0] * 100), 2),
    "%"
)

if len(pc1_peaks) >= 3:
    pc1_intervals = np.diff(pc1_peaks) / fs
    pc1_hr = 60 / np.median(pc1_intervals)
    pc1_cv = np.std(pc1_intervals) / np.mean(pc1_intervals)

    print("PC1 detected peaks:", len(pc1_peaks))
    print("PC1 estimated HR:", round(float(pc1_hr), 2), "BPM")
    print("PC1 interval CV:", round(float(pc1_cv), 3))
else:
    print("PC1: not enough peaks for a reliable preliminary HR estimate.")


# --- Preliminary comparison against ECG reference ---
ecg = df["ecg"].iloc[start:end].to_numpy(dtype=float)

if not np.all(np.isfinite(ecg)):
    print("\nECG comparison skipped: non-finite values.")
else:
    ecg_filtered = sosfiltfilt(
        butter(4, [5.0, 20.0], btype="bandpass",
               fs=fs, output="sos"),
        ecg - np.mean(ecg)
    )

    # ECG R-peak detection; threshold is a starting point
    r_peaks, _ = find_peaks(
        ecg_filtered,
        distance=int(0.4 * fs),
        prominence=0.5 * np.std(ecg_filtered)
    )

    if len(r_peaks) >= 3:
        ecg_intervals = np.diff(r_peaks) / fs
        ecg_hr = 60 / np.median(ecg_intervals)

        print("\n--- ECG Reference Comparison ---")
        print("Detected ECG peaks:", len(r_peaks))
        print("ECG estimated HR:", round(float(ecg_hr), 2), "BPM")
        print("PCA PPG estimated HR:", round(float(pc1_hr), 2), "BPM")
        print(
            "Absolute HR difference:",
            round(float(abs(ecg_hr - pc1_hr)), 2),
            "BPM"
        )
    else:
        print("\nECG comparison skipped: too few detected ECG peaks.")



