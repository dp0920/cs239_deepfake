# -*- coding: utf-8 -*-
import os
import subprocess
import argparse
from codecarbon import EmissionsTracker


def run_stylegan2_ada_pytorch_script(seeds):
    # Adjust this path and command to your StyleGAN2 setup
    print("Running StyleGAN2-ADA-PyTorch generation...")
    subprocess.run(
        [
            "python",
            "generate.py",
            "--outdir=out",
            "--trunc=1",
            "--seeds=" + seeds,
            "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl",
        ],
        cwd="stylegan2-ada-pytorch",
    )


def measure_emissions(task_function, task_name, *args):
    os.makedirs("./codecarbon_logs", exist_ok=True)
    tracker = EmissionsTracker(project_name=task_name, output_dir="./codecarbon_logs")
    tracker.start()
    try:
        task_function(*args)
    finally:
        emissions = tracker.stop()
        print(task_name + " emissions: {:.6}".format(emissions) + " kg CO₂")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure emissions for StyleGAN2-ADA-PyTorch generation."
    )
    parser.add_argument(
        "seeds", type=str, help="Seeds for image generation (e.g., '1-100')."
    )
    args = parser.parse_args()
    print("Starting Deepfake Energy Measurement...")

    # Measure StyleGAN2
    measure_emissions(
        run_stylegan2_ada_pytorch_script, "StyleGAN2-ADA-PyTorch-Generation", args.seeds
    )

    print("Measurement Complete.")
