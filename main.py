from huggingface_hub import snapshot_download
import torch
from PIL import Image
import sys
import os

# Add the 'stylegan2-pytorch' subdirectory to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "stylegan2-pytorch"))

# From `stylegan2-pytorch` directory
from model import Generator

STYLE_GAN_REPO = "ZeqiangLai/StyleGAN2"
STYLE_GAN_VERSION = (
    "bb4e00dc8ac0afbb9e0a5b6f72754c679673ee2e"  # Ensures we always get the same model
)
STYLE_GAN_FILE = "stylegan2-car-config-f.pt"

# Download model repo to a local directory
model_path = snapshot_download(repo_id=STYLE_GAN_REPO)
print("Model downloaded to:", model_path)

# Load generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = model_path + "/" + STYLE_GAN_FILE

G = Generator(size=1024, style_dim=512, n_mlp=8).cuda()

with open(model_path, "rb") as f:
    hf_weights = torch.load(f, map_location=device)["g_ema"]

G.load_state_dict(hf_weights["g_ema"])

# Generate random latent vector
z = torch.randn(1, G.z_dim).cuda()

# Generate image
img = G(z, None)  # (N, C, H, W) in [-1, 1] range

# Convert to image format (0-255 uint8)
img = (img.clamp(-1, 1) + 1) * (255 / 2)
img = img.permute(0, 2, 3, 1)  # (N, H, W, C)
img = img[0].cpu().numpy().astype("uint8")

Image.fromarray(img).save("output.png")
