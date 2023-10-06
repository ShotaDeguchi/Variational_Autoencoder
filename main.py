"""
Variational Autoencoder
reference: https://github.com/pytorch/examples/tree/main/vae
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

import train_utils
from models import *
from train_utils import *


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-e", "--epochs", type=int, default=10)
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument("-s", "--seed", type=int, default=42)


def main():
    # arguments
    args = parser.parse_args()

    # seed
    torch.manual_seed(args.seed)

    # device agnostic code
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    # create directories
    pathlib.Path("checkpoints").mkdir(parents=True, exist_ok=True)
    pathlib.Path("results").mkdir(parents=True, exist_ok=True)

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

    # visualize the dataset
    plt.figure(figsize=(7, 5))
    for i in range(5):
        image, label = train_dataset[i]
        plt.subplot(1, 5, i+1)
        plt.imshow(image.squeeze(), cmap="gray")
        plt.title(f"label: {label}")
    plt.tight_layout()
    plt.savefig(f"results/samples.png")
    plt.close()

    # dataloader
    num_workers = os.cpu_count()
    print(f"num_workers: {num_workers}")
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

    # model
    model = VariationalAutoencoder()
    print(summary(model, input_size=(args.batch_size, 1, 28, 28)))

    # training
    model.to(device)
    loss_fn = train_utils.loss_fn
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    hist_dict = {
        "epoch": [],
        "train_loss": [],
        "test_loss": []
    }
    for epoch in range(0, args.epochs+1):
        # train step
        train_loss, _ = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            accuracy_fn=None,
            device=device
        )

        # test step
        test_loss, _ = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=None,
            device=device
        )

        # add to history
        hist_dict["epoch"].append(epoch)
        hist_dict["train_loss"].append(train_loss)
        hist_dict["test_loss"].append(test_loss)

        # print the progress
        log = f"epoch: {epoch}/{args.epochs}, " \
                f"train_loss: {train_loss:.4f}, " \
                f"test_loss: {test_loss:.4f}"
        print(log)

        # reconstruct images
        #with torch.inference_mode():

        # generate images
        with torch.inference_mode():
            # sample from the latent space
            z = torch.randn(args.batch_size, 20).to(device)
            y = model.decode(z).cpu()

            # save the images
            save_image(
                y.view(args.batch_size, 1, 28, 28),
                f"results/sample_{epoch}.png"
            )

        # visualize the 2D latent space
        with torch.inference_mode():
            pass

    # training history
    plt.figure(figsize=(7, 5))
    plt.plot(hist_dict["epoch"], hist_dict["train_loss"], label="train_loss")
    plt.plot(hist_dict["epoch"], hist_dict["test_loss"], label="test_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/loss.png")
    plt.close()

    # save the model
    torch.save(model.state_dict(), "checkpoints/model.pt")


if __name__ == "__main__":
    main()
