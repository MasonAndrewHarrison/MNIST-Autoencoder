import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Encoder, Decoder
import matplotlib.pyplot as plt 
import numpy as np
import os



batch_size = 100
epochs = 50
learning_rate = 3e-4
latent_size = 80

device = "cuda" if torch.cuda.is_available() else "cpu"

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
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


encoder = Encoder(latent_size).to(device)
decoder = Decoder(latent_size).to(device)

opt_encoder = optim.Adam(encoder.parameters(), lr=learning_rate)
opt_decoder = optim.Adam(decoder.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

for epoch in range(epochs):
    for i, (image,_) in enumerate(loader):

        image = image.to(device)
        
        latent_space = encoder(image)
        decoded_image = decoder(latent_space)

        loss = criterion(image, decoded_image)
        
        opt_encoder.zero_grad()
        opt_decoder.zero_grad()

        loss.backward()

        opt_encoder.step()
        opt_decoder.step()

        if i == 0:
            
            torch.save(encoder.state_dict(), "Encoder_Weights.pth")
            torch.save(decoder.state_dict(), "Decoder_Weights.pth")

            print(loss.item())
            image = image.detach().to("cpu")

            plt.imshow(image[0, 0,:, :], cmap="gray")
            plt.show()


        