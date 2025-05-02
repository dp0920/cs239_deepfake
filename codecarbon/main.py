import os, sys, argparse
import subprocess
from codecarbon import EmissionsTracker

def get_git_root():
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        )
        return root.decode().strip()
    except subprocess.CalledProcessError:
        return None

def run_stylegan2_ada_pytorch_script(idx: int):
    # Adjust this path and command to your StyleGAN2 setup
    print("Running StyleGAN2-ADA-PyTorch generation...")
    git_root = get_git_root()
    subprocess.run([
        "python", "generate.py", "--outdir=out", "--trunc=1", "--idx=" + str(idx),
        "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl"
    ], cwd=git_root + "/codecarbon/stylegan2-ada-pytorch")

def measure_emissions(task_function, task_name, codecarbon_output, *args):
    os.makedirs(codecarbon_output, exist_ok=True)
    tracker = EmissionsTracker(project_name=task_name, output_dir=codecarbon_output)
    tracker.start()
    try:
        task_function(*args)
    finally:
        emissions = tracker.stop()
        print(f"{task_name} emissions: {emissions:.6f} kg CO₂")

if __name__ == "__main__":
    #print("Starting Deepfake Energy Measurement...")

    if len(sys.argv) not in [2, 3]:
        print("Usage: python your_script.py <idx> [<directory_for_emissions_csv>]")
        sys.exit(1)

    idx = int(sys.argv[1])
    codecarbon_output=sys.argv[2] if len(sys.argv) == 3 else "./codecarbon_logs"
        

    # Measure StyleGAN2
    measure_emissions(run_stylegan2_ada_pytorch_script, "StyleGAN2-ADA-PyTorch-Generation", codecarbon_output, idx)

    print("Measurement Complete.")
