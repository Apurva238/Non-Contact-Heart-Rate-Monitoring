# Part 2 — Frequency-Informed PCA rPPG Extraction

This package implements **Part 2** of the camera-based heart-rate monitoring project: it takes per-frame RGB skin-region signals from Part 1, applies PCA, chooses a component using frequency-informed scoring, bandpass-filters it, and exports a one-dimensional rPPG signal for Part 3.

## Files

- `pca_rppg.py` — main processing module and command-line interface.
- `requirements.txt` — Python dependencies.
- `examples/sample_rgb_signals.csv` — clearly labelled synthetic input for a smoke test; it is not real camera data.
- `tests/test_pca_rppg.py` — basic automated tests.

## Input contract from Part 1

CSV columns required:

| Column | Meaning |
|---|---|
| `timestamp_s` | Timestamp in seconds for each video frame |
| `red`, `green`, `blue` | Mean RGB values for the tracked skin region(s) |
| `valid` | Optional boolean-like flag; false rows are ignored |

RGB values may be raw 0–255 values or normalized values, provided each channel uses a consistent scale throughout the recording. The code estimates the sampling rate from timestamps unless `--fs` is provided. The input should represent one continuous recording; large gaps are interpolated, so inspect timestamp gaps and validity in the upstream module.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python pca_rppg.py examples/sample_rgb_signals.csv --output-dir output
```

For real Part 1 output:

```bash
python pca_rppg.py path/to/rgb_signals.csv --output-dir output
```

Optional parameters:

```bash
python pca_rppg.py rgb_signals.csv --output-dir output --low-bpm 42 --high-bpm 180
python pca_rppg.py rgb_signals.csv --output-dir output --fs 30
```

The high cutoff must be below the Nyquist frequency (half the sampling rate). At 30 frames/second, 180 BPM (3 Hz) is below the 15 Hz Nyquist limit.

## Outputs

- `processed_rppg.csv`: `timestamp_s`, normalized `rppg`, `valid`. This is the primary output for Part 3.
- `component_scores.csv`: score, peak frequency/BPM, in-band spectral peak ratio, autocorrelation support, explained variance, and selected flag for PC1–PC3.
- `metadata.json`: sampling rate, filter bounds, selected component, score rule, and run details.

## Method

1. Validate required columns and remove invalid RGB/timestamp rows.
2. Estimate sampling rate from the median positive timestamp difference, unless explicitly overridden.
3. Resample RGB channels onto a uniform time grid.
4. Linearly detrend and standardize the RGB channels.
5. Compute three PCA components using singular-value decomposition.
6. Bandpass-filter each component in the configured heart-rate band (default 42–180 BPM).
7. Score each component using `log(peak PSD / median in-band PSD) + max(0, autocorrelation at the peak period)` and select the highest-scoring component.
8. Normalize the selected filtered component and export it.

The method intentionally does **not** select PC1 automatically. PCA variance alone does not establish which component contains pulse.

## Important limitations

This is an initial engineering heuristic, not a clinically validated heart-rate estimator. A periodic head movement, expression change, or lighting fluctuation can also produce a strong spectral peak. Part 3 should report its own confidence/quality checks, and the complete system should be evaluated against reference heart-rate measurements. The included synthetic example only checks that the code runs; it does not validate extraction from real faces.

## Test

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```
