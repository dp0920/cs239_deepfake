import os, sys, argparse
from pathlib import Path
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

def run_biggan(n, seed_base, outdir=None):
    """
    Launch biggan-main.py to generate `n` images using seed_base,
    so each batch produces unique seeds.
    """

    print(f"Running BigGAN generation: n={n}, seed_base={seed_base}")
    codecarbon_dir = Path(__file__).parent.resolve()

    # 2) build the absolute path to biggan-main.py
    biggan_script = codecarbon_dir / "biggan-main.py"

    cmd = [
        sys.executable,
        str(biggan_script),
        "--images", str(n),
        "--seed-base", str(seed_base),
    ]

    if outdir is not None:
        cmd += ["--outdir", outdir]

    subprocess.run(cmd, check=True)


def measure_emissions(fn, task_name, outdir, *args):
    os.makedirs(outdir, exist_ok=True)
    tracker = EmissionsTracker(project_name=task_name, output_dir=outdir)
    tracker.start()
    try:
        fn(*args)
    finally:
        emissions = tracker.stop()
        print(f"{task_name} emissions: {emissions:.6f} kg CO₂")

if __name__ == "__main__":
    print("Starting Deepfake Energy Measurement...")
    if len(sys.argv) not in [3, 4]:
        print("Usage: python your_script.py <idx> [<directory_for_emissions_csv>]")
        sys.exit(1)

    model_type = sys.argv[1]

    if model_type == "stylegan":
        idx = int(sys.argv[2])
        codecarbon_output = sys.argv[3] if len(sys.argv) == 3 else "./codecarbon_logs"
        measure_emissions(run_stylegan2_ada_pytorch_script, "StyleGAN2-ADA-PyTorch-Generation", codecarbon_output, idx)

    elif model_type == "biggan":
        print("Starting Biggan Energy Measurement...")
        parser = argparse.ArgumentParser()
        parser.add_argument("images", type=int,
                            help="How many images to generate this run")
        parser.add_argument("csv_outdir", type=str,
                            help="Where to dump CodeCarbon logs")
        parser.add_argument("run_idx", type=int,
                            help="This run’s index (1…K)")

        args = parser.parse_args()

        print(f"Run #{args.run_idx}: Generating {args.images} images → logging in {args.csv_outdir}")
        measure_emissions(
            lambda: run_biggan(args.images, args.run_idx, args.csv_outdir),
            task_name="biggan",
            outdir=args.csv_outdir
        )


    print("Measurement Complete.")
