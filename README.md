# StyleGAN2-ADA-PyTorch: GPU-Based Image Generation

This project utilizes NVIDIA's StyleGAN2-ADA-PyTorch to generate high-quality images using pre-trained models. Follow the steps below to set up your environment and start generating images.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Activating Virtual Environment](#activating-virtual-environment)
- [Loading Modules](#loading-modules)
- [Optional: Clear PyTorch Cache](#optional-clear-pytorch-cache)
- [Running the Code](#running-the-code)
- [Generating Images with Pre-trained Model](#generating-images-with-pre-trained-model)

## Motivation
Modern ML models used for deepfake generation require a ton of compute power and energy. We present a study that quantifies the
energy use of two deepfake image generation
models - StyleGAN2 ADA and BigGAN - to
estimate the rate of carbon emissions during
model inference. We demonstrate that there is
a considerable impact on the environment when
using these large models and convey the ethical implications of continued usage. Furthermore, we found non-trivial variability between
GPUs and architectures when it comes to carbon emissions: for example, when generating
50 images over 100 batches with the BigGAN
model, there’s a difference of 0.347901 grams
of CO2 between the 256 and 512 versions of
the model. We discuss the need to research
more energy efficient innovations in the space
of AI development to protect the environment
for future generations.

## Environment Setup

1. **Create a Virtual Environments for each model**:

   ```bash
   
   # To run stylegan on v100 & p100
   python3 -m venv .venv368
   source .venv368/bin/activate
   pip install -r requirements-368.txt
   # and finally deactivate to install the other requirements
   deactivate
   
   # To run stylegan on a100
   python3 -m venv .venv310
   source .venv310/bin/activate
   pip install -r requirements-310.txt
   # and finally deactivate to install the other requirements
   deactivate
   
   # To run biggan
   python3 -m venv .bigganVenv
   source .bigganVenv/bin/activate
   pip install -r requirements-biggan.txt
   # and finally deactivate
   deactivate
   ```

## Loading Modules

To run these locally, you'll need CUDA and GCC, the batch runner script will take care of setting this up.

   ```bash
         module load cuda/11.7
         module load gcc/11.2.0
         export CXX=$(which g++)
         export CC=$(which gcc)
   ```

## Optional: Clear some cache

To remove cached PyTorch extensions (optional):

   ```bash
         rm -rf $HOME/.cache/torch_extensions
         pip cache purge
   ```

## Running the Code

To run the main script:

   ```bash
         cd codecarbon
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

## Batch Runs

There is a shell script, `sbatch_runner.sh` that can be run on SLURM environments for batching calls and resources 
requests. 

```shell
   GPU_COUNT=4
   # Bellow will create 6 jobs, 1 for each of the GPUs outlined in the paper, each configured with 4 GPUs.
   ./sbatch_runner.sh $GPU_COUNT
```
