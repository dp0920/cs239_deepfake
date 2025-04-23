from huggingface_hub import hf_hub_download,snapshot_download
from tensorflow.keras.models import load_model
import numpy as np
import torch
import PIL.Image
import sys

STYLE_GAN_REPO = "ZeqiangLai/StyleGAN2"
STYLE_GAN_VERSION = "bb4e00dc8ac0afbb9e0a5b6f72754c679673ee2e" # Ensures we always get the same model
STYLE_GAN_FILE = "stylegan2-car-config-f.pt"

# Download model repo to a local directory
model_path = snapshot_download(repo_id=STYLE_GAN_REPO, )
print(f"Model downloaded to: {model_path}")

# Load generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = f"{model_path}/{STYLE_GAN_FILE}"

sys.path.append(model_path)

from generator import StyleGAN2Generator

G = StyleGAN2Generator(size=1024, style_dim=512, n_mlp=8).to(device)

ckpt = torch.load(model_path, map_location=device)
G.load_state_dict(ckpt["g_ema"], strict=False)
G.eval()
"""
with open(f"{local_path}/{STYLE_GAN_FILE}", 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()
"""

# Generate latent vector
z = torch.from_numpy(np.random.randn(1, G.z_dim)).cuda()

# Generate image
img = G(z, None)[0]
img = (img.clamp(-1, 1) + 1) * (255 / 2)
img = img.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
PIL.Image.fromarray(img).save("fake_face.png")





if __name__ == '__main__':
    # Download ONNX face swap model from Hugging Face
    model_path = hf_hub_download(repo_id=STYLE_GAN_REPO, filename=STYLE_GAN_FILE)
    print(f"Model downloaded to: {model_path}")

    # Load the model
    model = load_model(model_path)

    # Generate latent vector and image
    z = np.random.randn(1, 512)
    generated = model.predict(z)

    # Postprocess and save
    generated = (generated[0] + 1) * 127.5  # if output is [-1, 1]
    img = np.clip(generated, 0, 255).astype(np.uint8)
    PIL.Image.fromarray(img).save("stylegan2_output.png")

