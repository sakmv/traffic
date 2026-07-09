import streamlit as st
import torch
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
import os

from diffusion import Diffusion, DiffusionConfig

st.set_page_config(page_title="DDPM Demo", layout="centered")

st.markdown("""
    <style>
    h1, h2, h3, .stCaption { text-align: center; }
    div.stButton { display: flex; justify-content: center; }
    div[data-testid="stRadio"] { display: flex; justify-content: center; }
    </style>
""", unsafe_allow_html=True)

st.title("Denoising Diffusion Probabilistic Model")
st.caption("Trained from scratch on Fashion-MNIST")


@st.cache_resource
def load_model():
    config = DiffusionConfig(image_size=32, in_channels=1)
    diffusion = Diffusion(config)
    diffusion.model.load_state_dict(
        torch.load("diffusion_model.pt", map_location=config.device, weights_only=True)
    )
    diffusion.model.eval()
    return diffusion, config


diffusion, config = load_model()
st.success(f"Model loaded on {config.device}")

if os.path.exists("loss_curve.png"):
    with st.expander("Training loss curve"):
        st.image("loss_curve.png", use_container_width=True)

st.divider()
st.header("Watch the denoising process")

mode = st.radio(
    "Display mode",
    ["Grid (snapshots)", "Live (single image)"],
    horizontal=True,
)

if mode == "Grid (snapshots)":
    n_samples = st.slider("Number of images to generate side-by-side", 1, 8, 4)

    steps_input = st.text_input(
        "Timesteps to snapshot (comma-separated, high = noisy, 0 = final image)",
        value=f"{config.timesteps},750,500,250,100,0"
    )

    if st.button("Generate with visible denoising steps", type="primary"):
        try:
            snapshot_steps = [int(s.strip()) for s in steps_input.split(",")]
        except ValueError:
            st.error("Couldn't parse the steps list — use comma-separated integers, e.g. 1000,500,0")
            st.stop()

        with st.spinner(f"Running reverse diffusion for {config.timesteps} steps..."):
            final_x, snapshots = diffusion.sample_with_intermediates(
                n_samples=n_samples, snapshot_steps=snapshot_steps
            )

        ordered_steps = sorted(snapshots.keys(), reverse=True)
        st.subheader("Denoising progression")
        for step in ordered_steps:
            grid = make_grid(snapshots[step], nrow=n_samples)
            img = grid.permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            label = "Pure noise (start)" if step == config.timesteps else (
                "Final image" if step == 0 else f"Step t = {step}"
            )
            st.image(Image.fromarray(img), caption=label, use_container_width=True)
    else:
        st.info("Set the step list above, then click generate to watch the images emerge from noise.")

else:  # Live mode
    if st.button("Generate live", type="primary"):
        left, center, right = st.columns([1, 3, 1])
        with center:
            placeholder = st.empty()

        x = torch.randn(
            (1, config.in_channels, config.image_size, config.image_size)
        ).to(config.device)

        diffusion.model.eval()
        with torch.no_grad():
            for i in reversed(range(1, config.timesteps)):
                t = (torch.ones(1) * i).long().to(config.device)
                predicted_noise = diffusion.model(x, t)

                alpha = diffusion.alpha[t][:, None, None, None]
                alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
                beta = diffusion.beta[t][:, None, None, None]

                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise

                if i % 20 == 0 or i == 1:
                    img = ((x.clamp(-1, 1) + 1) / 2)[0, 0].cpu().numpy()
                    img = (img * 255).astype(np.uint8)
                    with center:
                        placeholder.image(Image.fromarray(img), caption=f"t = {i}", use_container_width=True)

        diffusion.model.train()
        final_img = ((x.clamp(-1, 1) + 1) / 2)[0, 0].cpu().numpy()
        final_img = (final_img * 255).astype(np.uint8)
        with center:
            placeholder.image(Image.fromarray(final_img), caption="Final image", use_container_width=True)
    else:
        st.info("Click generate to watch a single image denoise live.")

with st.expander("How this works"):
    st.markdown("""
    1. **Forward process**: real images are progressively noised over `T` timesteps.
    2. **U-Net**: trained to predict the noise added at each timestep.
    3. **Reverse process**: starting from pure Gaussian noise, the model iteratively
       denoises step-by-step (`T` down to `0`) to produce a new image.
    """)