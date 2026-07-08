import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import math 
import torch.nn as nn

transform = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x:x*2 -1)
])
#compose string computings together. it will first to tensor and then run lambda to make the range -1 to 1

dataset = torchvision.datasets.FashionMNIST(root="./data",train=True,download=True,transform=transform)
#class fashionMNIST and then downloads in data and trains and download tru and transform calls transform

loader = torch.utils.data.DataLoader(dataset , batch_size=128, shuffle=True)
#makes the dataset batched of 128 images and shuffles
print(loader)
x,_=next(iter(loader))
print(x.shape,x.min().item(),x.max().item())

#Noise Schedule 
T_steps = 300
betas = torch.linspace(1e-4,0.02,T_steps)
alphas=1.0-betas
alphas_cumprod = torch.cumprod(alphas,dim=0)
print(alphas_cumprod)

#now, u have the % of image left after each iter
#but for 200 steps u would do 200 operations, instead we use direct forumula
#x1 = root| alpha prod x0|  +   root|1-(alpha prod * noise)|
# this is closed form formula
def forward_df(xt_1,t,alphas_cumprod):
    noise=torch.randn_like(xt_1)
    rt_alpha_bar=alphas_cumprod[t].sqrt().view(-1,1,1,1)
    rt_minus_alpha_bar=(1-alphas_cumprod[t]).sqrt().view(-1,1,1,1)
    xt= rt_alpha_bar*xt_1 + rt_minus_alpha_bar*noise
    return xt,noise

#checking if it works
x0 = x[0:1]  # one image
print(x)
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, t_val in enumerate([0, 50, 150, 250, 299]):
    t = torch.tensor([t_val])
    x_t, _ = forward_df(x0, t, alphas_cumprod)
    img = (x_t[0, 0] + 1) / 2  # back to [0,1] for display
    axes[i].imshow(img.detach().numpy(), cmap="gray")
    axes[i].set_title(f"t={t_val}")
    axes[i].axis("off")
plt.savefig("noise_progression.png")
plt.show()

class TimestepEmbedding(torch.nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim=dim
    
    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim) / half_dim).to(t.device)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding  

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x, t_emb):
        h = self.act(self.conv1(x))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act(self.conv2(h))
        return h

class SimpleUNet(nn.Module):
    def __init__(self, time_dim=64):
        super().__init__()
        self.time_embed = TimestepEmbedding(time_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.ReLU())

        self.down1 = Block(1, 32, time_dim)
        self.down2 = Block(32, 64, time_dim)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = Block(64, 64, time_dim)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.up_block1 = Block(64 + 64, 32, time_dim)
        self.up2 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.up_block2 = Block(32 + 32, 32, time_dim)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_embed(t))

        d1 = self.down1(x, t_emb)          # 28x28, 32ch
        d2 = self.down2(self.pool(d1), t_emb)  # 14x14, 64ch

        b = self.bottleneck(self.pool(d2), t_emb)  # 7x7, 64ch

        u1 = self.up1(b)                   # 14x14, 64ch
        u1 = self.up_block1(torch.cat([u1, d2], dim=1), t_emb)  # 14x14, 32ch

        u2 = self.up2(u1)                  # 28x28, 32ch
        u2 = self.up_block2(torch.cat([u2, d1], dim=1), t_emb)  # 28x28, 32ch

        return self.out(u2)                # 28x28, 1ch — predicted noise
    
model = SimpleUNet()
x_dummy = torch.randn(8, 1, 28, 28)
t_dummy = torch.randint(0, T_steps, (8,))
out = model(x_dummy, t_dummy)
print(out.shape)  # should be [8, 1, 28, 28]