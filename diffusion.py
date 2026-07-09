
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class DiffusionConfig:
    image_size: int = 32
    in_channels: int = 1
    base_channels: int = 64
    time_emb_dim: int = 256
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """Simple Conv -> GroupNorm -> GELU block"""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.GroupNorm(8, out_ch)
        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.GELU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class SimpleUNet(nn.Module):
    """A minimal U-Net to predict noise"""
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        image_channels = config.in_channels
        down_channels = (64, 128, 256, 512)
        up_channels = (512, 256, 128, 64)
        out_dim = config.in_channels 
        time_emb_dim = config.time_emb_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU()
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) \
                                    for i in range(len(down_channels)-1)])
        
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) \
                                  for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
            
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
            
        return self.output(x)

class Diffusion(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.model = SimpleUNet(config).to(config.device)
        
        self.beta = torch.linspace(config.beta_start, config.beta_end, config.timesteps).to(config.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        ε = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * ε, ε

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.config.timesteps, size=(n,), device=self.config.device)

    def forward(self, x):
        t = self.sample_timesteps(x.shape[0])
        x_t, noise = self.noise_images(x, t)
        predicted_noise = self.model(x_t, t)
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def sample(self, n_samples):
        self.model.eval()
        x = torch.randn((n_samples, self.config.in_channels, self.config.image_size, self.config.image_size)).to(self.config.device)
        
        for i in reversed(range(1, self.config.timesteps)):
            t = (torch.ones(n_samples) * i).long().to(self.config.device)
            predicted_noise = self.model(x, t)
            
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            
        self.model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        return x
    
    @torch.no_grad()
    def sample_with_intermediates(self, n_samples=1, x_start=None, snapshot_steps=None): 
        self.model.eval()
        x = torch.randn(
            (n_samples, self.config.in_channels, self.config.image_size, self.config.image_size)
          ).to(self.config.device)

        if snapshot_steps is None:
          snapshot_steps = [self.config.timesteps, self.config.timesteps // 2, 0]
        snapshot_steps = set(snapshot_steps)

        snapshots = {}
        if self.config.timesteps in snapshot_steps:
          snapshots[self.config.timesteps] = ((x.clamp(-1, 1) + 1) / 2).cpu()

        for i in reversed(range(1, self.config.timesteps)):
          t = (torch.ones(n_samples) * i).long().to(self.config.device)
          predicted_noise = self.model(x, t)

          alpha = self.alpha[t][:, None, None, None]
          alpha_hat = self.alpha_hat[t][:, None, None, None]
          beta = self.beta[t][:, None, None, None]

          noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

          x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

          if i in snapshot_steps:
              snapshots[i] = ((x.clamp(-1, 1) + 1) / 2).cpu()

        self.model.train()
        final_x = (x.clamp(-1, 1) + 1) / 2

        if 0 in snapshot_steps:
          snapshots[0] = final_x.cpu()

        return final_x, snapshots 
