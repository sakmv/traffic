import torch
import matplotlib.pyplot as plt
import time
from main import SimpleUNet,betas,alphas,alphas_cumprod,forward_df, T_steps,loader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = SimpleUNet().to(device)
betas = betas.to(device)
alphas = alphas.to(device)
alphas_cumprod = alphas_cumprod.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_step(x0, model, alphas_cumprod, T_steps, optimizer):
    x0 = x0.to(device)
    batch_size = x0.shape[0]

    t = torch.randint(0, T_steps, (batch_size,), device=device)
    x_t, true_noise = forward_df(x0, t, alphas_cumprod)

    predicted_noise = model(x_t, t)
    loss = torch.nn.functional.mse_loss(predicted_noise, true_noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

epochs = 20
loss_history = []

for epoch in range(epochs):
    start = time.time()
    total_loss = 0

    for x0, _ in loader:
        loss = train_step(x0, model, alphas_cumprod, T_steps, optimizer)
        total_loss += loss

    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} | loss: {avg_loss:.4f} | time: {time.time()-start:.1f}s")

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("loss_curve.png")
plt.show()
torch.save(model.state_dict(), "diffusion_model.pt")



















































































































































# ---- sampling function ----
@torch.no_grad()
def sample(model, alphas, alphas_cumprod, betas, T_steps, img_shape=(1,1,28,28)):
    x = torch.randn(img_shape, device=device)

    for t in reversed(range(T_steps)):
        t_batch = torch.tensor([t], device=device)
        predicted_noise = model(x, t_batch)

        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        beta = betas[t]

        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        x = (1/alpha.sqrt()) * (x - ((1-alpha)/(1-alpha_bar).sqrt())*predicted_noise) + beta.sqrt()*noise

    return x

# ---- generate and view one sample ----
generated = sample(model, alphas, alphas_cumprod, betas, T_steps)
img = (generated[0,0].cpu() + 1) / 2
plt.imshow(img, cmap="gray")
plt.title("Generated sample")
plt.savefig("first_sample.png")
plt.show()