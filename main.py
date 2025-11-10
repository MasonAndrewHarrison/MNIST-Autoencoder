import torch
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib.widgets import Slider, Button
from model import Decoder, Encoder
import random



latents = torch.load("latents.pt")

latent_size = 80
latent_amount = len(latents)

device = "cuda" if torch.cuda.is_available() else "cpu"
decoder = Decoder(latent_size).to(device)
encoder = Encoder(latent_size).to(device)

decoder.load_state_dict(torch.load("Decoder_Weights.pth", map_location=device))
encoder.load_state_dict(torch.load("Encoder_Weights.pth", map_location=device))

decoder.eval()
encoder.eval()

z_noise = latents[random.randint(1, latent_amount),: ,:, :].unsqueeze(0)
image = decoder(z_noise).view(64, 64).detach().cpu()

fig, ax = plt.subplots(figsize=(21, 15))
plt.subplots_adjust(bottom=0.25, left=0.6)

img_display = ax.imshow(image, cmap="gray")
slider_list = [0]*latent_size


for i in range(8):
    for j in range(10):

        true_index = j+(i*10)
        print(true_index)
        ax_z = plt.axes([0.03 + (j/20), 0.9 - (i/10.5), 0.03, 0.067])
        slider_list[true_index] = Slider(ax_z, true_index+1, valmin=-4.0, valmax=4.0, valinit=z_noise[0, true_index, 0, 0].item(), orientation="vertical")

button_ax1 = plt.axes([.65, .25, .2, .05])
button1 = Button(
    button_ax1,
    "Reset Z Noise",
)
button_ax2 = plt.axes([.65, .15, .2, .05])
button2 = Button(
    button_ax2,
    "New Latent Sample",
)

def update(val, index):
    z_noise[0, index, 0, 0] = torch.tensor(val, device=device)
    with torch.no_grad():
        new_image = decoder(z_noise).view(64, 64).detach().cpu()
    img_display.set_data(new_image)
    fig.canvas.draw_idle()


def on_button_click_sample(event):

    global z_noise
    z_noise = latents[random.randint(1, latent_amount),: ,:, :].unsqueeze(0)
    
    for i in range(latent_size):
        slider_list[i].set_val(z_noise[0, i, 0, 0].item())

    fig.canvas.draw_idle()

def on_button_click_random(event):

    global z_noise
    z_noise = torch.randn(1, latent_size, 1, 1).to(device)
    
    for i in range(latent_size):
        slider_list[i].set_val(z_noise[0, i, 0, 0].item())

    fig.canvas.draw_idle()


for i in range(latent_size):

    slider_list[i].on_changed(lambda x, idx=i: update(x, idx))


button1.on_clicked(on_button_click_random)
button2.on_clicked(on_button_click_sample)
plt.show()