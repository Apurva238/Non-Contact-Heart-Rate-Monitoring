import numpy as np
import pandas as pd
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent

INPUT_CSV = (
    ROOT.parent
    / "video_processing"
    / "output"
    / "rgb_signals.csv"
)

OUTPUT_CSV = (
    ROOT
    / "input"
    / "rgb_forehead_chrominance.csv"
)

# Load original RGB data
df = pd.read_csv(INPUT_CSV)

rgb_columns = [
    "forehead_red",
    "forehead_green",
    "forehead_blue",
]

rgb = (
    df[rgb_columns]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype=float)
)

timestamps = df["timestamp_s"].to_numpy(dtype=float)

# Read the forehead validity flags.
valid = (
    df["forehead_valid"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1", "yes"])
    .to_numpy()
)

# Exclude invalid or non-finite measurements.
valid = (
    valid
    & np.isfinite(timestamps)
    & np.all(np.isfinite(rgb), axis=1)
    & (rgb.sum(axis=1) > 1e-12)
)

# Normalize each RGB triplet by its total intensity.
chrominance = np.full_like(rgb, np.nan)

chrominance[valid] = (
    rgb[valid]
    / rgb[valid].sum(axis=1, keepdims=True)
)

# Save in Pranjal's expected input format.
output = pd.DataFrame({
    "timestamp_s": timestamps,
    "red": chrominance[:, 0],
    "green": chrominance[:, 1],
    "blue": chrominance[:, 2],
    "valid": valid,
})

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

output.to_csv(
    OUTPUT_CSV,
    index=False
)

print("Normalized RGB input created.")
print("Total frames:", len(output))
print("Valid frames:", int(valid.sum()))
print("Output:", OUTPUT_CSV)