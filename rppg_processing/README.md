# rPPG Signal Processing

This module implements the signal-processing stages of a non-contact heart-rate monitoring system using facial videos.

It processes normalized facial RGB signals, extracts a representative rPPG waveform using Principal Component Analysis (PCA), and estimates heart rate using frequency-domain analysis.

## Processing Pipeline

1. **RGB Preprocessing:** Normalizes RGB signals extracted from the forehead region.
2. **PCA:** Applies Principal Component Analysis to the RGB signals.
3. **rPPG Extraction:** Selects the first principal component (PC1) as the representative rPPG signal.
4. **Heart Rate Estimation:** Uses Welch's Power Spectral Density (PSD) method to identify the dominant frequency and calculate heart rate.

## Files

- `prepare_forehead_chrominance.py` – Prepares normalized forehead RGB signals.
- `pca_rppg.py` – Extracts the rPPG signal using PCA.
- `estimate_hr_now.py` – Estimates heart rate from the extracted rPPG signal.
- `heart_rate_estimator.py` – Heart-rate estimation functionality.

## Running the Module

Run the following commands from the project root directory:

```bash
python rppg_processing/prepare_forehead_chrominance.py

python rppg_processing/pca_rppg.py rppg_processing/input/rgb_forehead_chrominance.csv --output-dir rppg_processing/output/forehead_chrominance

python rppg_processing/estimate_hr_now.py
```

Alternatively, execute the complete project using:

```bash
python run_pipeline.py
```

## Output Files

The processing pipeline generates:

- `rgb_forehead_chrominance.csv` – Preprocessed forehead RGB signals.
- `processed_rppg.csv` – Extracted one-dimensional rPPG waveform.
- `component_scores.csv` – PCA component information.
- `metadata.json` – Processing metadata.

## Heart Rate Estimation

Welch's Power Spectral Density method estimates the dominant frequency of the extracted rPPG signal.

Heart rate is calculated as:

\[
HR = 60 \times f
\]

where \(f\) is the dominant frequency in Hz.

The current implementation searches within a predefined range of 85–110 BPM, selected for the sample recording.

## Limitations

- PCA does not guarantee that PC1 contains only pulse-related information.
- Lighting variations and facial movements can affect signal quality.
- The current heart-rate search range limits general applicability.
- Further validation using multiple recordings and reference measurements is required.