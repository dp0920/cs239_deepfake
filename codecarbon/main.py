import os, sys, argparse
import subprocess
from codecarbon import EmissionsTracker

def run_stylegan2_ada_pytorch_script():
    # Adjust this path and command to your StyleGAN2 setup
    print("Running StyleGAN2-ADA-PyTorch generation...")
    subprocess.run([
        "python", "generate.py", "--outdir=out", "--trunc=1", "--seeds=1-100",
        "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl"
    ], cwd="stylegan2-ada-pytorch")

def measure_emissions(task_function, task_name):
    os.makedirs("./codecarbon_logs", exist_ok=True)
    tracker = EmissionsTracker(project_name=task_name, output_dir="./codecarbon_logs")
    tracker.start()
    try:
        task_function()
    finally:
        emissions = tracker.stop()
        print(f"{task_name} emissions: {emissions:.6f} kg CO₂")

if __name__ == "__main__":
    print("Starting Deepfake Energy Measurement...")

    # Measure StyleGAN2
    measure_emissions(run_stylegan2_ada_pytorch_script, "StyleGAN2-ADA-PyTorch-Generation")

    print("Measurement Complete.")

