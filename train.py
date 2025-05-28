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

import models
import train_utils
from models import *
from train_utils import *

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--f_out", type=int, default=2)
parser.add_argument("-b", "--batch_size", type=int, default=1024)
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument("-e", "--epochs", type=int, default=30)
parser.add_argument("-p", "--patience", type=int, default=10)
parser.add_argument("-d", "--device", type=str, default="mps")
parser.add_argument("-s", "--seed", type=int, default=42)
args = parser.parse_args()

################################################################################

def main():
    # seed
    torch.manual_seed(args.seed)

    # device agnostic code
    device = args.device
    print(f"device: {device}")

    # create directories
    path_checkpoints = pathlib.Path("checkpoints")
    path_results_train = pathlib.Path("results_train")
    pathlib.Path("checkpoints").mkdir(parents=True, exist_ok=True)
    pathlib.Path("results_train").mkdir(parents=True, exist_ok=True)

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
    plt.savefig(path_results_train / "dataset.png")
    plt.close()

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

    # training
    model.to(device)
    loss_fn = train_utils.loss_fn
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate
    )
    hist_dict = {
        "epoch": [],
        "train_loss": [],
        "train_bc_etr": [],
        "train_kl_div": [],
        "test_loss": [],
        "test_bc_etr": [],
        "test_kl_div": [],
    }
    wait = 0
    best_loss = np.inf
    t0 = time.perf_counter()
    for epoch in range(0, args.epochs+1):
        # train step
        train_loss, train_bc_etr, train_kl_div, _ = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            accuracy_fn=None,
            device=device
        )

        # test step
        test_loss, test_bc_etr, test_kl_div, _ = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=None,
            device=device
        )

        # add to history
        hist_dict["epoch"].append(epoch)
        hist_dict["train_loss"].append(train_loss)
        hist_dict["train_bc_etr"].append(train_bc_etr)
        hist_dict["train_kl_div"].append(train_kl_div)

        hist_dict["test_loss"].append(test_loss)
        hist_dict["test_bc_etr"].append(test_bc_etr)
        hist_dict["test_kl_div"].append(test_kl_div)

        # early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"early stopping at epoch: {epoch}")
                break

        # print the progress
        t1 = time.perf_counter()
        elps = t1 - t0
        log = f"epoch: {epoch}/{args.epochs}, " \
                f"train_loss: {train_loss:.3f}, " \
                f"test_loss: {test_loss:.3f}, " \
                f"best_loss: {best_loss:.3f}, " \
                f"wait: {wait}/{args.patience}, " \
                f"elps: {elps:.3f}s"
        print(log)

        # reconstruct images
        with torch.inference_mode():
            # sample from test dataset
            torch.manual_seed(args.seed)
            n_samples = 8
            images, labels = next(iter(test_dataloader))   # first batch -> (batch_size, 1, 28, 28)
            images = images[:n_samples].to(device)   # first several images -> (n_samples, 1, 28, 28)

            # reconstruct the images
            images_, mu_, sigma_ = model(images.view(-1, 784))
            images = images.cpu()
            images_ = images_.cpu()

            # save the images
            images = images.view(n_samples, 1, 28, 28)
            images_ = images_.view(n_samples, 1, 28, 28)
            images = torch.cat([images, images_], dim=0)
            save_image(
                images,
                path_results_train / f"reconstructed_{epoch}.png"
            )

        # generate images
        with torch.inference_mode():
            # sample from the latent space
            torch.manual_seed(args.seed)
            n_samples = 64
            z = torch.randn(n_samples, args.f_out).to(device)
            y = model.decode(z).cpu()

            # save the images
            save_image(
                y.view(n_samples, 1, 28, 28),
                path_results_train / f"generated_{epoch}.png"
            )

        # visualize the 2D embedding of the latent space
        with torch.inference_mode():
            # sample from test dataset
            torch.manual_seed(args.seed)
            images, labels = next(iter(test_dataloader))   # first batch -> (batch_size, 1, 28, 28)
            images, labels = images.to(device), labels.to(device)

            # encode the images
            mu, sigma = model.encode(images.view(-1, 784))
            z = model.reparameterize(mu, sigma)
            mu, sigma, z = mu.cpu(), sigma.cpu(), z.cpu()
            images, labels = images.cpu(), labels.cpu()

            # plot
            fig, ax = plt.subplots(figsize=(5, 5))
            for i in range(10):
                ax.scatter(z[labels==i, 0], z[labels==i, 1], label=str(i), marker="o", alpha=.7)
                ax.annotate(
                    str(i), (z[labels==i, 0].mean(), z[labels==i, 1].mean()),
                    bbox=dict(boxstyle="round", fc="w", ec="k", alpha=.7)
                )
            ax.legend(loc="upper left")
            ax.set(
                xlim=(-7., 7.),
                ylim=(-7., 7.),
                xlabel=r"$z_1$",
                ylabel=r"$z_2$",
                title=f"2D embedding of the latent space at epoch: {epoch}",
            )
            fig.tight_layout()
            fig.savefig(path_results_train / f"embedding_{epoch}.png")
            plt.close(fig)


    # history
    nrows, ncols = 2, 1
    fig, ax = plt.subplots(2, 1, figsize=(7, 10))

    ax[0].plot(hist_dict["epoch"], hist_dict["train_loss"], c="b", label="train_loss")
    ax[0].plot(hist_dict["epoch"], hist_dict["test_loss"],  c="r", label="test_loss")

    ax[1].plot(hist_dict["epoch"], hist_dict["train_bc_etr"], ls="--", c="b", label="train_bc_etr")
    ax[1].plot(hist_dict["epoch"], hist_dict["train_kl_div"], ls="-.", c="b", label="train_kl_div")
    ax[1].plot(hist_dict["epoch"], hist_dict["test_bc_etr"],  ls="--", c="r", label="test_bc_etr")
    ax[1].plot(hist_dict["epoch"], hist_dict["test_kl_div"],  ls="-.", c="r", label="test_kl_div")

    for i in range(nrows):
        ax[i].legend(loc="best")
        ax[i].set(
            yscale="log",
            xlabel=r"Epochs",
            ylabel=r"$\mathcal{L}$",
        )
    fig.tight_layout()
    fig.savefig(path_results_train / "history.png")
    plt.close(fig)


    # save the model
    torch.save(
        model.state_dict(),
        path_checkpoints / "model.pt"
    )

################################################################################

def plot_settings():
    plt.style.use("default")
    # plt.style.use("seaborn-v0_8-deep")
    plt.style.use("seaborn-v0_8-talk")   # paper / notebook / talk / poster
    # plt.style.use("classic")
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = True
    plt.rcParams['axes.axisbelow'] = True   # background grid
    plt.rcParams["grid.alpha"] = .3
    plt.rcParams["legend.framealpha"] = .8
    plt.rcParams["legend.facecolor"] = "w"
    plt.rcParams["savefig.dpi"] = 300

################################################################################

if __name__ == "__main__":
    plot_settings()
    main()
