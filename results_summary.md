# PPG Signal Processing and Synthetic Signal Generation

## Method
A real PPG recording (`s1_sit.csv`) was analyzed at a sampling frequency of 500 Hz. PPG signals were bandpass-filtered between 0.5 and 5 Hz, and heart rate was estimated from detected pulse peaks. ECG was filtered between 5 and 20 Hz and used as a reference. PCA fusion was evaluated using four PPG channels: `pleth_1`, `pleth_2`, `pleth_5`, and `pleth_6`.

## Preliminary validation
Five non-overlapping 20-second segments were evaluated.

- `pleth_2` mean absolute heart-rate error against ECG: 0.24 BPM.
- PCA fusion mean absolute heart-rate error against ECG: 0.19 BPM.
- PCA performed slightly better on these five tested segments.
- These results are preliminary and do not establish general or clinical accuracy.

## Synthetic PPG generator
A normalized template derived from real PPG cycles was used to generate periodic synthetic PPG signals at specified heart rates, sampling frequencies, and durations. Tests at 60, 75, 90, and 120 BPM verified the expected sample counts and beat timing.

## Limitations
The synthetic signal repeats a normalized template and does not model all physiological variation or motion artifacts. Validation used a small number of segments from one recording. ECG peak detections and performance on additional recordings require further evaluation.

## Interface
`generate_clean_ppg(heart_rate, sampling_frequency, duration)` returns the time array, synthetic PPG signal, and metadata.
