import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Settings
FILE = "s1_sit.csv"
FS = 500
SEGMENT_SECONDS = 20
SEGMENT_LENGTH = FS * SEGMENT_SECONDS
STARTS = [1000, 11000, 21000, 31000, 41000]
PPG_CHANNELS = ["pleth_1", "pleth_2", "pleth_5", "pleth_6"]

df = pd.read_csv(FILE)
print("Dataset loaded:", df.shape)

# Bandpass filters
ppg_sos = butter(
    4, [0.5, 5], btype="bandpass", fs=FS, output="sos"
)
ecg_sos = butter(
    4, [5, 20], btype="bandpass", fs=FS, output="sos"
)


def estimate_hr(signal, min_prominence_factor=0.5):
    """Estimate HR and beat-interval variability from a filtered signal."""
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)
    prominence = min_prominence_factor * np.std(signal)

    candidates = []

    # Try both polarities because signal peaks may point up or down.
    for candidate_signal in (signal, -signal):
        peaks, _ = find_peaks(
            candidate_signal,
            distance=int(0.4 * FS),
            prominence=prominence
        )

        if len(peaks) < 3:
            continue

        intervals = np.diff(peaks) / FS
        hr = 60 / np.median(intervals)
        cv = np.std(intervals) / np.mean(intervals)

        if 40 <= hr <= 180:
            candidates.append((len(peaks), cv, hr))

    if not candidates:
        return np.nan, np.nan, 0

    # Prefer a plausible detection with more beats; use lower CV to break ties.
    candidates.sort(key=lambda x: (-x[0], x[1]))
    count, cv, hr = candidates[0]
    return hr, cv, count


results = []

for start in STARTS:
    end = start + SEGMENT_LENGTH
    segment = df.iloc[start:end]

    if len(segment) != SEGMENT_LENGTH:
        print(f"Skipping segment at row {start}: not enough data")
        continue

    # Skip segments that cross a timestamp discontinuity, if detectable.
    timestamps = pd.to_datetime(segment["time"], errors="coerce")
    if timestamps.isna().any() or (timestamps.diff().dropna() <= pd.Timedelta(0)).any():
        print(f"Skipping segment at row {start}: invalid or non-increasing timestamps")
        continue

    # Filter ECG and estimate reference HR.
    ecg = sosfiltfilt(ecg_sos, segment["ecg"].to_numpy(dtype=float))
    ecg_hr, _, ecg_peaks = estimate_hr(ecg)

    # Filter PPG channels.
    filtered = {}
    for channel in PPG_CHANNELS:
        filtered[channel] = sosfiltfilt(
            ppg_sos, segment[channel].to_numpy(dtype=float)
        )

    # Individual PPG baseline: pleth_2.
    ppg2_hr, ppg2_cv, ppg2_peaks = estimate_hr(filtered["pleth_2"])

    # PCA fusion: standardize the selected channels, then use PC1.
    matrix = np.column_stack([filtered[ch] for ch in PPG_CHANNELS])
    matrix = StandardScaler().fit_transform(matrix)
    pc1 = PCA(n_components=1).fit_transform(matrix).ravel()
    pca_hr, pca_cv, pca_peaks = estimate_hr(pc1)

    results.append({
        "Start row": start,
        "ECG HR": ecg_hr,
        "pleth_2 HR": ppg2_hr,
        "pleth_2 error": abs(ppg2_hr - ecg_hr)
            if np.isfinite(ppg2_hr) and np.isfinite(ecg_hr) else np.nan,
        "PCA HR": pca_hr,
        "PCA error": abs(pca_hr - ecg_hr)
            if np.isfinite(pca_hr) and np.isfinite(ecg_hr) else np.nan,
        "ECG peaks": ecg_peaks,
        "PPG peaks": ppg2_peaks,
        "PCA peaks": pca_peaks,
    })

if not results:
    print("No valid segments were processed. Check the CSV and timestamps.")
else:
    report = pd.DataFrame(results)
    print("\n--- Segment Validation Results ---")
    print(report.round(2).to_string(index=False))

    print("\n--- Mean Absolute HR Error vs ECG (BPM) ---")
    print("pleth_2:", round(report["pleth_2 error"].mean(), 2))
    print("PCA:    ", round(report["PCA error"].mean(), 2))

    report.to_csv("validation_results.csv", index=False)
    print("\nSaved results to validation_results.csv")