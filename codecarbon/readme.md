# StyleGAN2-ADA-PyTorch: GPU-Based Image Generation

This project utilizes NVIDIA's StyleGAN2-ADA-PyTorch to generate high-quality images using pre-trained models. Follow the steps below to set up your environment and start generating images.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Activating Virtual Environment](#activating-virtual-environment)
- [Loading Modules](#loading-modules)
- [Optional: Clear PyTorch Cache](#optional-clear-pytorch-cache)
- [Running the Code](#running-the-code)
- [Generating Images with Pre-trained Model](#generating-images-with-pre-trained-model)

## Environment Setup

1. **Create a Virtual Environment**:

   ```bash
   python3 -m venv .venv
   ```

2. **Activate the Virtual Environment**:

   ```bash
   source .venv/bin/activate
   ```

3. **Install Required Packages**:

   ```bash
   pip install codecarbon torch Pillow ninja
   ```

## Activating Virtual Environment

Ensure that the virtual environment is activated before proceeding:

```bash
source .venv/bin/activate
```

## Loading Modules

Load the necessary modules for CUDA and GCC:

```bash
module load cuda/11.7
module load gcc/11.2.0
export CXX=$(which g++)
export CC=$(which gcc)
```

## Optional: Clear PyTorch Cache

To remove cached PyTorch extensions (optional):

```bash
rm -rf $HOME/.cache/torch_extensions
```

## Running the Code

To run the main script:

```bash
python main.py
```

This script will execute the StyleGAN2-ADA-PyTorch generation process and measure the associated carbon emissions using CodeCarbon.

## Generating Images with Pre-trained Model

Alternatively, you can generate images directly using the `generate.py` script:

```bash
cd stylegan2-ada-pytorch
python generate.py --outdir=out --trunc=1 --seeds=2 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
```

- `--outdir=out`: Directory to save the generated images.
- `--trunc=1`: Truncation psi value; controls the trade-off between variety and fidelity.
- `--seeds=2`: Seed value(s) for image generation.
- `--network=...`: URL or path to the pre-trained model.

---

For more details and advanced configurations, refer to the [official StyleGAN2-ADA-PyTorch repository](https://github.com/NVlabs/stylegan2-ada-pytorch).


