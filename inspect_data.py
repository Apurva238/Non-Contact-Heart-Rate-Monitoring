import pandas as pd

# Load the dataset

df = pd.read_csv("s1_sit.csv")

# Convert timestamps

df["time"] = pd.to_datetime(df["time"])

# Basic information

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# Estimate sampling rate, ignoring backward timestamp jumps

time_diff = df["time"].diff().dt.total_seconds()
valid_intervals = time_diff[time_diff > 0]

print("\nMedian sampling interval:", valid_intervals.median(), "seconds")
print("Estimated sampling rate:", 1 / valid_intervals.median(), "Hz")
print("Valid intervals:", len(valid_intervals))


# Check missing values in PPG channels

ppg_columns = [f"pleth_{i}" for i in range(1, 7)]
print("\nMissing values in PPG channels:")
print(df[ppg_columns].isna().sum())

print("\nFirst 5 timestamps:")
print(df["time"].head().to_list())

print("\nLast 5 timestamps:")
print(df["time"].tail().to_list())

print("\nTimestamps increasing:", df["time"].is_monotonic_increasing)
print("Timestamps decreasing:", df["time"].is_monotonic_decreasing)

# Find places where timestamps stop increasing

time_diff = df["time"].diff().dt.total_seconds()

backward_jumps = df.loc[time_diff < 0, ["time"]].copy()
backward_jumps["previous_time"] = df["time"].shift(1)[time_diff < 0]
backward_jumps["difference_seconds"] = time_diff[time_diff < 0]

print("\nNumber of backward timestamp jumps:", len(backward_jumps))
print("\nFirst 10 backward jumps:")
print(backward_jumps.head(10))

import matplotlib.pyplot as plt

sampling_rate = 500

# Select 5 seconds from the first recording segment

start = 1000
end = start + 5 * sampling_rate

fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)

for i in range(1, 7):
        column = f"pleth_{i}"
        signal = df[column].iloc[start:end].to_numpy()
        time = range(len(signal))

        axes[i - 1].plot([t / sampling_rate for t in time], signal)
        axes[i - 1].set_ylabel(column)
        axes[i - 1].grid(True)

axes[-1].set_xlabel("Time (seconds)")
fig.suptitle("Six PPG Channels — 5-Second Segment")
plt.tight_layout()
plt.show()

