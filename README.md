# Non-Contact Heart Rate Monitoring from Face Videos

A Linear Systems and Signal Processing (LSSP) project that estimates heart rate from facial videos using Remote Photoplethysmography (rPPG) and digital signal processing techniques.

The system extracts subtle colour variations from facial skin, processes them to obtain an rPPG signal, and estimates heart rate in beats per minute (BPM).

## Project Workflow

The complete implementation follows this pipeline:

```text
Input Facial Video
       |
       v
Face Detection and Tracking
       |
       v
Facial Region Extraction
       |
       v
RGB Signal Extraction
       |
       v
Chrominance Preprocessing
       |
       v
Principal Component Analysis
       |
       v
rPPG Signal Extraction
       |
       v
Welch's Power Spectral Density
       |
       v
Heart Rate Estimation (BPM)
```

### 1. Video Processing
- Reads the input facial video frame by frame.
- Detects faces using OpenCV Haar Cascade.
- Tracks the detected face across consecutive frames.

### 2. Region of Interest (ROI) Extraction
- Identifies forehead, left cheek, and right cheek regions.
- Applies skin masking to select suitable facial pixels.
- Uses the forehead region for the current heart-rate estimation pipeline.

### 3. RGB Signal Extraction
- Calculates average red, green, and blue intensities from facial skin regions.
- Creates RGB time-series signals from consecutive video frames.

### 4. Preprocessing
- Normalizes the forehead RGB signals.
- Prepares the colour information for PCA-based signal extraction.

### 5. PCA-Based rPPG Extraction
- Applies Principal Component Analysis (PCA) to the processed RGB signals.
- Selects the first principal component (PC1) as the representative rPPG signal.

### 6. Heart Rate Estimation
- Uses Welch's Power Spectral Density (PSD) method to analyse the extracted rPPG signal.
- Identifies the dominant frequency.
- Converts the frequency into beats per minute.

The conversion is:

\[
HR = 60 \times f
\]

where \(f\) is the dominant frequency in Hz.

## Technologies Used

- Python 3.12
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
├── run_pipeline.py
├── results_summary.md
│
├── Video_processing/
│   ├── video_rgb_extractor.py
│   ├── data/
│   │   └── subject1/
│   │       ├── vid.mp4
│   │       └── ground_truth.txt
│   └── output/
│
└── rppg_processing/
    ├── README.md
    ├── prepare_forehead_chrominance.py
    ├── pca_rppg.py
    ├── estimate_hr_now.py
    ├── heart_rate_estimator.py
    ├── input/
    └── output/
```

Generated intermediate files and local environment files are not required to be committed to the repository.

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

## Input Video

Place the input facial video at:

```text
Video_processing/data/subject1/vid.mp4
```

The current implementation expects the video at this location.

## Running the Project

Run the complete pipeline from the project root:

```bash
python run_pipeline.py
```

The master script automatically performs:

1. Facial RGB extraction.
2. Forehead chrominance preprocessing.
3. PCA-based rPPG extraction.
4. Heart-rate estimation.

The estimated heart rate is displayed in the terminal.

## Sample Result

For the sample video used during development:

```text
Method: PCA-based rPPG
Estimated Heart Rate: 94.92 BPM
```

This result corresponds to the tested recording and is not a guaranteed result for other videos.

## Experimental Details

| Parameter | Value |
|---|---:|
| Video duration | 57 seconds |
| Frame rate | 30 FPS |
| Total frames | 1710 |
| Resolution | 1920 × 1080 |
| Face detection | 1710/1710 frames |
| Valid forehead frames | 1710 |
| Selected PCA component | PC1 |
| Estimated heart rate | 94.92 BPM |

## Limitations

- Performance depends on lighting conditions, head movement, camera quality, and skin visibility.
- Changes in facial colour are subtle and can be affected by external disturbances.
- PCA does not guarantee that the selected component contains only pulse-related information.
- The current heart-rate estimator searches within a predefined range of 85–110 BPM.
- The implementation requires further evaluation across different subjects, videos, and lighting conditions.
- This project is an academic signal-processing demonstration and is not intended for medical diagnosis.

## Future Scope

- Explore alternative rPPG extraction techniques such as POS and CHROM.
- Improve motion and illumination artifact removal.
- Extend the supported heart-rate range.
- Evaluate performance using multiple subjects and recordings.
- Compare estimates against synchronized reference measurements.

## Acknowledgements

- OpenCV for computer vision functionality.
- NumPy, SciPy, and Pandas for scientific computing and signal processing.

## License

No license has been selected for this project yet.