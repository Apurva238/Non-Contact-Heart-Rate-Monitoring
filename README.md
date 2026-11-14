# Non-Contact Heart Rate Monitoring from Face Videos

A Linear Systems and Signal Processing (LSSP) project exploring Photoplethysmography (PPG) and Remote Photoplethysmography (rPPG) for heart-rate monitoring.

The project consists of two main components:

1. **Synthetic PPG Generation:** Generates physiologically inspired PPG signals and studies the effects of noise and disturbances.
2. **Video-Based Heart Rate Estimation:** Extracts facial colour variations from video and processes them using PCA-based rPPG signal extraction.

## Project Workflow

### 1. Synthetic PPG Generation

- Uses a representative PPG heartbeat template derived from a PhysioNet recording.
- Generates synthetic PPG signals for a specified heart rate, sampling frequency, and duration.
- Simulates noise and disturbances to study their effects on the signal.
- Supports waveform visualization and validation.

### 2. Video-Based rPPG Estimation

The current implementation follows this pipeline:

1. **Video Processing:** Reads the input video frame by frame.
2. **Face Detection:** Detects the face using OpenCV.
3. **ROI Extraction:** Extracts RGB signals from the forehead region.
4. **Preprocessing:** Prepares the RGB signals using chrominance-based preprocessing.
5. **PCA:** Extracts the rPPG signal using Principal Component Analysis.
6. **Heart Rate Estimation:** Uses Welch's Power Spectral Density method to estimate the dominant pulse frequency and convert it into BPM.

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- OpenCV
- Principal Component Analysis (PCA)
- Welch's Power Spectral Density (PSD)

## Repository Structure

```text
LSSP_Project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── analyse_ppg.py
├── noise_simulation.py
├── ppg_generator.py
├── ppg_pipeline.py
├── ppg_template.npy
├── run_full_pipeline.py
├── validate_segments.py
├── visualization_validation.py
│
├── Video_processing/
│   ├── video_rgb_extractor.py
│   └── data/
│       └── subject1/
│           ├── vid.mp4
│           └── ground_truth.txt
│
├── rppg_processing/
│   ├── prepare_forehead_chrominance.py
│   ├── pca_rppg.py
│   ├── estimate_hr_now.py
│   ├── heart_rate_estimator.py
│   └── signal_quality.py
│
└── outputs/
    ├── synthetic_ppg/
    └── rppg/
```

Generated intermediate files and local input data are excluded from the Git repository.

## Installation

### Prerequisites

- Python 3.12 recommended
- pip

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Create a virtual environment

Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Video-Based Heart Rate Estimation

Place the input video at:

```text
Video_processing/data/subject1/vid.mp4
```

Run the following commands from the project root, in order:

**Step 1: Extract facial RGB signals**

```bash
python Video_processing/video_rgb_extractor.py
```

**Step 2: Prepare forehead chrominance signals**

```bash
python rppg_processing/prepare_forehead_chrominance.py
```

**Step 3: Extract the rPPG signal using PCA**

```bash
python rppg_processing/pca_rppg.py rppg_processing/input/rgb_forehead_chrominance.csv --output-dir rppg_processing/output/forehead_chrominance
```

**Step 4: Estimate heart rate**

```bash
python rppg_processing/estimate_hr_now.py
```

The estimated heart rate is displayed in BPM.

## Sample Result

For the sample video used during development:

```text
Method: PCA-based rPPG
Estimated Heart Rate: 94.92 BPM
```

This is the result obtained from the tested recording and is not a guaranteed result for other videos.

## Dataset

The synthetic PPG component was developed using the **Pulse Transit Time PPG** dataset from PhysioNet.

The source recording used during development:

- File: `s1_sit.csv`
- Sampling frequency: approximately 500 Hz

The raw dataset should be obtained from its original source rather than committed to this repository.

## Limitations

- rPPG performance depends on lighting, movement, camera quality, and face detection.
- The current video implementation uses the forehead region and PCA-based signal extraction.
- The current heart-rate estimator searches within 85–110 BPM, a range selected for the sample recording. It must be adapted and validated for recordings with different heart rates.
- The project is an academic signal-processing demonstration and is not intended for medical diagnosis.

## Acknowledgements

- PhysioNet for the Pulse Transit Time PPG dataset.
- The developers of the open-source scientific computing and computer vision libraries used in this project.

## License

No license has been selected for this project yet.