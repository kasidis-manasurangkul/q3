import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
from PIL import Image
import io

# Settings
latent_dim = 100
num_classes = 10
img_shape = (1, 28, 28)
device = torch.device('cpu')  # Streamlit Cloud does not provide GPU

# Define Generator class (should match your trained model's architecture)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, self.label_emb(labels)), -1)
        img = self.model(x)
        img = img.view(img.size(0), *img_shape)
        return img

# Load generator model
@st.cache_resource
def load_generator():
    model = Generator().to(device)
    model.load_state_dict(torch.load("generator.pth", map_location=device))
    model.eval()
    return model

generator = load_generator()

st.title("Handwritten Digit Generator (MNIST-style)")

# UI: Select digit
digit = st.selectbox("Select a digit to generate:", list(range(10)), index=0)

if st.button("Generate 5 Images"):
    with torch.no_grad():
        z = torch.randn(5, latent_dim, device=device)
        labels = torch.full((5,), digit, dtype=torch.long, device=device)
        gen_imgs = generator(z, labels)
        gen_imgs = gen_imgs * 0.5 + 0.5  # [-1,1] to [0,1]

        # Make a grid of images for nice display (1x5 row)
        grid = make_grid(gen_imgs, nrow=5, padding=2)
        # Convert to numpy and PIL Image for Streamlit
        ndarr = grid.mul(255).byte().cpu().numpy().transpose(1, 2, 0).squeeze()
        pil_img = Image.fromarray(ndarr)

        st.image(pil_img, caption=f'Generated images of {digit}', use_column_width=True)
