import subprocess
import sys


def run_step(description, command):
    print(f"\n{'=' * 55}")
    print(description)
    print('=' * 55)

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\nERROR: {description} failed.")
        sys.exit(result.returncode)


def main():

    print("\n")
    print("=" * 55)
    print("   NON-CONTACT HEART RATE MONITORING PIPELINE")
    print("=" * 55)

    # Step 1: Extract RGB signals from facial video
    run_step(
        "STEP 1: Facial RGB Extraction",
        [
            sys.executable,
            "Video_processing/video_rgb_extractor.py"
        ]
    )

    # Step 2: Prepare forehead chrominance signals
    run_step(
        "STEP 2: Chrominance Preprocessing",
        [
            sys.executable,
            "rppg_processing/prepare_forehead_chrominance.py"
        ]
    )

    # Step 3: Extract rPPG signal using PCA
    run_step(
        "STEP 3: PCA-Based rPPG Extraction",
        [
            sys.executable,
            "rppg_processing/pca_rppg.py",
            "rppg_processing/input/rgb_forehead_chrominance.csv",
            "--output-dir",
            "rppg_processing/output/forehead_chrominance"
        ]
    )

    # Step 4: Estimate heart rate
    run_step(
        "STEP 4: Heart Rate Estimation",
        [
            sys.executable,
            "rppg_processing/estimate_hr_now.py"
        ]
    )

    print("\n")
    print("=" * 55)
    print("   PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 55)


if __name__ == "__main__":
    main()
