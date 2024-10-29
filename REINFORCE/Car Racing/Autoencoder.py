import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1),  # 96 -> 48
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2, 1),  # 48 -> 24
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),  # 24 -> 12
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 12 -> 6
            nn.ReLU()
        )
        # add nn.Flatten() and dense layer later

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),  # 6 -> 12
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1),  # 12 -> 24
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, 2, 1, output_padding=1),  # 24 -> 48
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 3, 2, 1, output_padding=1),  # 48 -> 96
            nn.ReLU()
        )

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # self.unbottlenecker = nn.Linear(300, 64*6*6)

    def forward(self, x):
        encoded = self.encoder(x)
        # unbottlenecked = self.unbottlenecker(encoded)
        # decoder_input = unbottlenecked.reshape((1, 64, 6, 6))
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)



