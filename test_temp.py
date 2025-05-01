import os
import torch
from musetalk.models.vae import VAE

vae = VAE(
    model_path=os.path.join("models", 'sd-vae'),
)
vae_model = vae.vae
vae_model.eval()
vae_model.requires_grad_(False)
# 创建一个形状为 (250, 3, 256, 256) 的随机浮点数张量
tensor = torch.rand(25, 3, 256, 256, dtype=torch.float32).to(device='cuda')
masked_latents = vae_model.encode(tensor).latent_dist.mode()
print(tensor)