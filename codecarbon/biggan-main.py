import os
import torch
import argparse
import numpy as np

from pathlib import Path
import torch.backends.cudnn as cudnn
from torchvision.transforms import ToPILImage
from torch.nn.utils import remove_spectral_norm
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, one_hot_from_int


torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

""" Different kinds of classes directly comparable between ImageNET (BigGAN) and AFHQ (Animal faces)
AFHQ has three classes: dogs, cats, and wildlife
ImageNET has different kinds of dogs, cats, and foxes, cheetahs, wolves, leopard, tiger (which exist in AFHQ)"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1.  Load generator (256-px, pre-trained on ImageNet)
G = BigGAN.from_pretrained("biggan-deep-512").to(device).eval()


def generate_image(seed, trunc=0.4, outdir="biggan_samples"):
    rng = np.random.RandomState(seed)
    z = truncated_noise_sample(1, 128, truncation=trunc, seed=rng.randint(1e9))
    z = torch.tensor(z, device=device, dtype=torch.float32)

    # pick a class (e.g. fixed or from a list)
    y = torch.zeros((1, 1000), device=device)

    # The second argument is where the class index for the image goes. 248 = Husky images in ImageNET
    y[0, 248] = 1.0
    with torch.no_grad():
        img = G(z, y, truncation=trunc)[0]

    img = (img.clamp(-1, 1) + 1) / 2
    Path(outdir).mkdir(exist_ok=True)
    path = Path(outdir) / f"sample_{seed:05d}.png"

    ToPILImage()(img.cpu()).save(str(path))
    print(f"New image saved to {path}!")


def run_biggan(n=8, seed_base=1, outdir="biggan_samples"):
    """
    Generates `n` images with seeds:
       seed = (seed_base-1)*n + j
    so across runs you get unique seeds 0 … K*n-1.
    """
    for j in range(n):
        seed = (seed_base - 1) * n + j
        generate_image(seed=seed, outdir=outdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images", type=int, default=8, help="How many images this run"
    )
    parser.add_argument(
        "--seed-base", type=int, default=1, help="Batch index to offset seeds"
    )
    parser.add_argument(
        "--outdir", type=str, default="biggan_samples", help="Where to save PNGs"
    )

    args = parser.parse_args()

    run_biggan(n=args.images, seed_base=args.seed_base, outdir=args.outdir)


if __name__ == "__main__":
    main()
