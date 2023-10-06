"""
model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    def __init__(self, f_in=784, f_hid=512, f_out=20):
        super().__init__()

        # encoder
        self.fc1 = nn.Linear(f_in, f_hid)
        self.fc_mu = nn.Linear(f_hid, f_out)   # mu
        self.fc_sigma = nn.Linear(f_hid, f_out)    # sigma

        # decoder
        self.fc2 = nn.Linear(f_out, f_hid)
        self.fc3 = nn.Linear(f_hid, f_in)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.fc2(z))
        z = torch.sigmoid(self.fc3(z))
        return z

    def forward(self, x):
        x = x.view(-1, 784)
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        y = self.decode(z)
        return y, mu, sigma


