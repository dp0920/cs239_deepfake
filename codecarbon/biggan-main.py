from pathlib import Path
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from pytorch_pretrained_biggan import (
    BigGAN, truncated_noise_sample, one_hot_from_int
)

torch.set_grad_enabled(False)          # no autograd needed
device = 'cuda' if torch.cuda.is_available() else 'cpu'

''' Different kinds of classes directly comparable between ImageNET (BigGAN) and AFHQ (Animal faces)
AFHQ has three classes: dogs, cats, and wildlife
ImageNET has different kinds of dogs, cats, and foxes, cheetahs, wolves, leopard, tiger (which exist in AFHQ)'''
class_indices = [248, 153, 200, 229, 230, 254, 283, 281, 284, 287, 277, 293, 270, 288, 292]


# 1.  Load generator (256-px, pre-trained on ImageNet)
G = BigGAN.from_pretrained('biggan-deep-256').to(device).eval()

# 2.  Helper: generate n unconditional images
def biggan_random(n=8, trunc=0.4, outdir='biggan_samples', seed=None):
    Path(outdir).mkdir(exist_ok=True)
    rng = np.random.RandomState(seed)
    for k in range(len(class_indices)):
        # latent z  – sampled from N(0,1) then truncated
        z = truncated_noise_sample(1, 128, truncation=trunc, seed=rng.randint(1e9))
        z = torch.tensor(z, dtype=torch.float32, device=device)
        # random ImageNet label (0–999) in one-hot
        class_idx = class_indices[k]
        y = torch.zeros((1, 1000), dtype=torch.float32, device=device)
        y[0, class_idx] = 1.0

        # 3.  Forward pass
        with torch.no_grad():
            img = G(z, y, truncation=trunc)[0]     # shape (3,256,256), range [-1,1]
        img = (img.clamp(-1, 1) + 1) / 2           # → [0,1]

        ToPILImage()(img.cpu()).save(f'{outdir}/sample_{k:02d}.png')
        print(f'saved {outdir}/sample_{k:02d}.png  (class {class_idx})')

biggan_random(n=8, trunc=0.4, seed=123)
