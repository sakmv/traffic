
import torch
from diffusion import Diffusion, DiffusionConfig
from torchvision.utils import make_grid, save_image

config = DiffusionConfig(image_size=32, in_channels=1)
diffusion = Diffusion(config)
diffusion.model.load_state_dict(torch.load("diffusion_model.pt", map_location=config.device, weights_only=True))
diffusion.model.eval()

samples = diffusion.sample(n_samples=16)
grid = make_grid(samples, nrow=4)
save_image(grid, "samples.png")
print("Saved samples.png")
