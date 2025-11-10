import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(

            nn.Conv2d(1, 4, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),

            self._block(4, 8, 3, 2, 1),
            self._block(8, 16, 3, 2, 1),
            self._block(16, 32, 3, 2, 1),
            self._block(32, 64, 3, 2, 1),
            
            nn.Conv2d(64, latent_size, 4, 1, 0, bias=True)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_size):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 64, 2, 1, 0, bias=True),
            nn.ReLU(inplace=True),

            self._block(64, 32, 4, 2, 1),
            self._block(32, 16, 4, 2, 1),
            self._block(16, 8, 4, 2, 1),
            self._block(8, 4, 4, 2, 1),
            self._block(4, 2, 4, 2, 1),

            nn.ConvTranspose2d(2, 1, 3, 1, 1, bias=True),
            nn.Tanh()

        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.decoder(x)


if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.randn(1, 1, 64, 64).to(device)

    encoder = Encoder(100).to(device)
    decoder = Decoder(100).to(device)

    print(tensor.shape)
    new_tensor = encoder(tensor)
    print(new_tensor.shape)

    new_image = decoder(new_tensor)
    print(new_image.shape)



