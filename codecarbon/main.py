import os, sys, argparse
import subprocess
import os, argparse
from pathlib import Path
from codecarbon import EmissionsTracker

def run_stylegan2_ada_pytorch_script(count: int):
    # Adjust this path and command to your StyleGAN2 setup
    print("Running StyleGAN2-ADA-PyTorch generation...")
    subprocess.run([
        "python", "generate.py", "--outdir=out", "--trunc=1", "--count=" + str(count),
        "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl"
    ], cwd="codecarbon/stylegan2-ada-pytorch")

def measure_emissions(task_function, task_name, *args):
    os.makedirs("./codecarbon_logs", exist_ok=True)
    tracker = EmissionsTracker(project_name=task_name, output_dir="./codecarbon_logs")
    tracker.start()
    try:
        task_function(*args)
    finally:
        emissions = tracker.stop()
        print(f"{task_name} emissions: {emissions:.6f} kg CO2")
        
def main():
    print("Starting Biggan Energy Measurement...")
    parser = argparse.ArgumentParser()
    parser.add_argument("images",    type=int,
                        help="How many images to generate this run")
    parser.add_argument("csv_outdir",type=str,
                        help="Where to dump CodeCarbon logs")
    parser.add_argument("run_idx",   type=int,
                        help="This run’s index (1…K)")
                        
    args = parser.parse_args()
    
    print(f"Run #{args.run_idx}: Generating {args.images} images → logging in {args.csv_outdir}")

    if len(sys.argv) < 2:
        print("Usage: python your_script.py <count>")
        sys.exit(1)

    count = int(sys.argv[1])

    # Measure StyleGAN2
    measure_emissions(run_stylegan2_ada_pytorch_script, "StyleGAN2-ADA-PyTorch-Generation", count)

    print("Measurement Complete.")
        

if __name__ == "__main__":
    main()
