from pathlib import Path
import json
import subprocess
import sys
import numpy as np

from ppg_pipeline import generate_ppg_with_noise


ROOT = Path(__file__).resolve().parent


def run_script(name, script, *args):
    print(f"\n{name}")
    print("-" * 50)

    subprocess.run(
        [sys.executable, str(ROOT / script), *map(str, args)],
        cwd=ROOT,
        check=True
    )

    print(f"\nCompleted: {name}")


def main():

    print("\n========================================")
    print("     COMPLETE LSSP PROJECT PIPELINE")
    print("========================================")

    # ==========================================
    # PART 1: SYNTHETIC PPG GENERATION
    # ==========================================

    print("\n========== PART 1: SYNTHETIC PPG ==========")

    heart_rate = 75
    sampling_frequency = 100
    duration = 10

    output_dir = ROOT / "full_pipeline_output"
    output_dir.mkdir(exist_ok=True)

    print("\n[1/7] Generating clean and noisy PPG signals")

    time_array, signals, metadata = generate_ppg_with_noise(
        heart_rate=heart_rate,
        sampling_frequency=sampling_frequency,
        duration=duration
    )

    np.savetxt(
        output_dir / "clean_ppg.csv",
        np.column_stack((time_array, signals["clean"])),
        delimiter=",",
        header="time,clean_ppg",
        comments=""
    )

    np.savetxt(
        output_dir / "noisy_ppg.csv",
        np.column_stack((time_array, signals["combined"])),
        delimiter=",",
        header="time,noisy_ppg",
        comments=""
    )

    with open(output_dir / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print("Clean and noisy PPG signals generated successfully.")

    # ==========================================
    # PART 2: VISUALIZATION AND VALIDATION
    # ==========================================

    print("\n[2/7] Visualizing and validating PPG signals")

    run_script(
        "PPG Visualization and Validation",
        "visualization_validation.py",
        "--clean", output_dir / "clean_ppg.csv",
        "--noisy", output_dir / "noisy_ppg.csv",
        "--metadata", output_dir / "metadata.json",
        "--fs", sampling_frequency,
        "--outdir", output_dir / "visualization_results"
    )

    # ==========================================
    # PART 3: VIDEO-BASED HEART RATE ESTIMATION
    # ==========================================

    print("\n========== PART 2: VIDEO-BASED rPPG ==========")

    video = ROOT / "Video_processing/data/subject1/vid.mp4"

    if not video.exists():
        raise FileNotFoundError(
            f"Input video not found: {video}"
        )

    # Step 3: RGB extraction
    print("\n[3/7] Extracting facial RGB signals")

    run_script(
        "Facial RGB Extraction",
        "Video_processing/video_rgb_extractor.py"
    )

    # Step 4: Chrominance preprocessing
    print("\n[4/7] Preparing forehead chrominance signals")

    run_script(
        "Forehead Chrominance Preprocessing",
        "rppg_processing/prepare_forehead_chrominance.py"
    )

    # Step 5: PCA processing
    print("\n[5/7] Applying PCA-based rPPG processing")

    run_script(
        "PCA-based rPPG Extraction",
        "rppg_processing/pca_rppg.py",
        "rppg_processing/input/rgb_forehead_chrominance.csv",
        "--output-dir",
        "rppg_processing/output/forehead_chrominance"
    )

    # Step 6: Heart rate estimation
    print("\n[6/7] Estimating heart rate")

    run_script(
        "Heart Rate Estimation",
        "rppg_processing/estimate_hr_now.py"
    )

    # ==========================================
    # COMPLETION
    # ==========================================

    print("\n========================================")
    print("       COMPLETE PIPELINE FINISHED!")
    print("========================================")

    print("\nGenerated results:")
    print("- Synthetic PPG: full_pipeline_output/")
    print("- Video RGB: Video_processing/output/")
    print("- Processed rPPG: rppg_processing/output/")
    print("- Final heart-rate estimate: displayed above")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nPIPELINE FAILED: {error}")
        sys.exit(1)