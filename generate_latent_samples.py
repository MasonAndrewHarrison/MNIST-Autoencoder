import torch
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib.widgets import Slider, Button
from model import Decoder, Encoder

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.0, 0.0),
        scale=(1.0, 1.0),
    ),
    transforms.Normalize((0.5,),(0.5,)),
])


dataset = datasets.MNIST(root="./data", transform=transform, download=True)

latent_size = 80
latent_list = []

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = Encoder(latent_size).to(device)
encoder.load_state_dict(torch.load("Encoder_Weights.pth", map_location=device))
encoder.eval()

for i, (real_image,_) in enumerate(dataset):
    if i > 1000: break
    with torch.no_grad():
        latent_space = encoder(real_image.unsqueeze(0).to(device))
    latent_list.append(latent_space)

all_latents = torch.cat(latent_list, dim=0)
torch.save(all_latents, "latents.pt")
