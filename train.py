# train.py
import os
import time
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from diffusion import Diffusion, DiffusionConfig

# train.py

# --- data ---
transform = transforms.Compose([
    transforms.Pad(2),                      # 28x28 -> 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

# --- model ---
config = DiffusionConfig(image_size=32, in_channels=1)
diffusion = Diffusion(config)
optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=1e-3)

print("Using device:", config.device)

# --- training loop ---
epochs = 50
loss_history = []
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(epochs):
    start = time.time()
    total_loss = 0

    for x0, _ in loader:
        x0 = x0.to(config.device)

        optimizer.zero_grad()
        loss = diffusion(x0)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} | loss: {avg_loss:.4f} | time: {time.time()-start:.1f}s")

# --- plot + save loss curve ---
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("loss_curve.png")

# --- save checkpoint ---
torch.save(diffusion.model.state_dict(), "diffusion_model.pt")
print("Saved diffusion_model.pt")
