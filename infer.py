"""
inference
generate images from the latent space
"""

import os
import pathlib
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchinfo import summary

import models
import train_utils
from models import *
from train_utils import *


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--f_out", type=int, default=2)
parser.add_argument("-b", "--batch_size", type=int, default=1024)
parser.add_argument("-e", "--epochs", type=int, default=100)
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument("-p", "--patience", type=int, default=10)
parser.add_argument("-d", "--device", type=str, default="cpu")
parser.add_argument("-s", "--seed", type=int, default=42)


def main():
    # arguments
    args = parser.parse_args()

    # plot settings
    plot_settings()

    # seed
    torch.manual_seed(args.seed)

    # device agnostic code
    device = args.device
    print(f"device: {device}")

    # create directories
    path_results_infer = pathlib.Path("results_infer")
    pathlib.Path("results_infer").mkdir(parents=True, exist_ok=True)

    # dataset
    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        transform=transforms.ToTensor(),
        target_transform=None,
        download=True
    )
    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        transform=transforms.ToTensor(),
        target_transform=None,
        download=True
    )
    print(f"train_dataset: {len(train_dataset)}")
    print(f"test_dataset: {len(test_dataset)}")

    # dataloader
    num_workers = os.cpu_count()
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    print(f"batch_size: {args.batch_size}")
    print(f"train_dataloader: {len(train_dataloader)}")
    print(f"test_dataloader: {len(test_dataloader)}")

    # model
    model = VariationalAutoencoder(f_out=args.f_out)
    print(summary(model, input_size=(args.batch_size, 1, 28, 28)))
    model.load_state_dict(torch.load("checkpoints/model.pt"))

    # freeze for safety
    for param in model.parameters():
        param.requires_grad = False

    # sample from latent space
    with torch.inference_mode():
        # sample from normal distribution
        torch.manual_seed(args.seed)
        zmin, zmax, z_n = -3., 3., 8
        z1 = torch.linspace(zmin, zmax, z_n)
        z2 = torch.linspace(zmin, zmax, z_n)
        z1, z2 = torch.meshgrid(z1, z2, indexing="ij")
        z1, z2 = z1.flatten(), z2.flatten()
        z = torch.stack([z1, z2], dim=1)
        y = model.decode(z)
        y = y.view(-1, 1, 28, 28)
        save_image(y, path_results_infer / "generated.png", nrow=z_n)

        # walk through latent space (z1_s, z2_s) -> (z1_e, z2_e)
        torch.manual_seed(args.seed)
        z_s = torch.tensor([-2, 2.])   # start
        z_e = torch.tensor([2, -2.])   # end
        z_n = 64
        z1s = torch.linspace(z_s[0], z_e[0], z_n)
        z2s = torch.linspace(z_s[1], z_e[1], z_n)
        for i in range(z_n):
            z = torch.tensor([z1s[i], z2s[i]])
            y = model.decode(z)
            y = y.view(-1, 1, 28, 28)
            save_image(y, path_results_infer / f"walk_{i}.png")

        # sweep through latent space
        torch.manual_seed(args.seed)
        zmin, zmax, z_n = -2., 2., 8
        z1 = torch.linspace(zmin, zmax, z_n)
        z2 = torch.linspace(zmin, zmax, z_n)
        for i, z1_ in enumerate(z1):
            z1_ = torch.ones_like(z2) * z1_
            z = torch.stack([z1_, z2], dim=1)
            y = model.decode(z)
            y = y.view(-1, 1, 28, 28)
            save_image(y, path_results_infer / f"sweep_along_z1_{i}.png", nrow=z_n)
        for i, z2_ in enumerate(z2):
            z2_ = torch.ones_like(z1) * z2_
            z = torch.stack([z1, z2_], dim=1)
            y = model.decode(z)
            y = y.view(-1, 1, 28, 28)
            save_image(y, path_results_infer / f"sweep_along_z2_{i}.png", nrow=z_n)


def plot_settings():
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["legend.framealpha"] = 1.
    plt.rcParams["savefig.dpi"] = 300


if __name__ == "__main__":
    main()
