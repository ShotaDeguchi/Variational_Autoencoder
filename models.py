"""
model implementation
"""

import torch
import torch.nn as nn


class VariationalAutoencoder(nn.Module):
    def __init__(self, f_in=784, f_out=2):
        super().__init__()

        # encoder
        self.fc_enc1 = nn.Linear(f_in, 256)
        self.fc_enc2 = nn.Linear(256, 64)
        self.fc_mu = nn.Linear(64, f_out)
        self.fc_sigma = nn.Linear(64, f_out)

        # decoder
        self.fc_dec1 = nn.Linear(f_out, 64)
        self.fc_dec2 = nn.Linear(64, 256)
        self.fc_dec3 = nn.Linear(256, f_in)

    def encode(self, x):
        x = nn.ReLU()(self.fc_enc1(x))
        x = nn.ReLU()(self.fc_enc2(x))
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = nn.ReLU()(self.fc_dec1(z))
        z = nn.ReLU()(self.fc_dec2(z))
        z = torch.sigmoid(self.fc_dec3(z))
        return z

    def forward(self, x):
        x = x.view(-1, 784)
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        y = self.decode(z)
        return y, mu, sigma