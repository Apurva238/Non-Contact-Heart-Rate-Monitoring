from pathlib import Path
import json
import subprocess
import sys
import numpy as np

from ppg_pipeline import generate_ppg_with_noise


def main():
    # 1. Configuration
    heart_rate = 75
    sampling_frequency = 100
    duration = 10

    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "full_pipeline_output"
    output_dir.mkdir(exist_ok=True)

    # 2. Generate clean and disturbed PPG signals
    time_array, signals, metadata = generate_ppg_with_noise(
        heart_rate=heart_rate,
        sampling_frequency=sampling_frequency,
        duration=duration,
    )

    # 3. Save clean PPG signal
    np.savetxt(
        output_dir / "clean_ppg.csv",
        np.column_stack((time_array, signals["clean"])),
        delimiter=",",
        header="time,clean_ppg",
        comments="",
    )

    # 4. Save noisy PPG signal
    np.savetxt(
        output_dir / "noisy_ppg.csv",
        np.column_stack((time_array, signals["combined"])),
        delimiter=",",
        header="time,noisy_ppg",
        comments="",
    )

    # 5. Save metadata
    with open(output_dir / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print("PPG signals generated and saved successfully.")

    # 6. Run Hitansh's visualization and validation script
    visualization_script = project_dir / "visualization_validation.py"

    subprocess.run(
        [
            sys.executable,
            str(visualization_script),
            "--clean", str(output_dir / "clean_ppg.csv"),
            "--noisy", str(output_dir / "noisy_ppg.csv"),
            "--metadata", str(output_dir / "metadata.json"),
            "--fs", str(sampling_frequency),
            "--outdir", str(output_dir / "visualization_results"),
        ],
        check=True,
    )

    print("\nFull pipeline completed!")
    print(f"Results folder: {output_dir}")


if __name__ == "__main__":
    main()