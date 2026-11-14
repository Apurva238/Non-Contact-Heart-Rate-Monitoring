import subprocess
import sys
from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parent

# Pipeline steps
steps = [
    {
        "name": "Extracting facial RGB signals",
        "command": [
            "Video_processing/video_rgb_extractor.py"
        ]
    },
    {
        "name": "Preparing forehead chrominance signals",
        "command": [
            "rppg_processing/prepare_forehead_chrominance.py"
        ]
    },
    {
        "name": "Applying PCA-based rPPG processing",
        "command": [
            "rppg_processing/pca_rppg.py",
            "rppg_processing/input/rgb_forehead_chrominance.csv",
            "--output-dir",
            "rppg_processing/output/forehead_chrominance"
        ]
    },
    {
        "name": "Estimating heart rate",
        "command": [
            "rppg_processing/estimate_hr_now.py"
        ]
    }
]

def main():
    print("\n========================================")
    print("  NON-CONTACT HEART RATE MONITORING")
    print("========================================")

    video = ROOT / "Video_processing/data/subject1/vid.mp4"

    if not video.exists():
        print("\nERROR: Input video not found!")
        print(f"Expected location: {video}")
        return

    for index, step in enumerate(steps, start=1):
        print(f"\n[{index}/{len(steps)}] {step['name']}")
        print("-" * 45)

        command = [sys.executable] + step["command"]

        try:
            subprocess.run(command, cwd=ROOT, check=True)
            print(f"\nCompleted: {step['name']}")

        except subprocess.CalledProcessError:
            print(f"\nERROR: Pipeline stopped at step {index}.")
            return

    print("\n========================================")
    print("       PIPELINE COMPLETED!")
    print("========================================")


if __name__ == "__main__":
    main()