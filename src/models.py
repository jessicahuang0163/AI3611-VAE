import torch
from torch import nn
from utils.base.base_block import *


class MLPEncoder(nn.Module):
    def __init__(self, exp_specs):
        super(MLPEncoder, self).__init__()

        self.img_size = exp_specs["img_size"]
        self.latent_dim = exp_specs["latent_dim"]
        self.net = nn.Sequential(
            nn.Linear(self.img_size ** 2, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(16, 2 * self.latent_dim),
        )

    def forward(self, batch):
        return torch.chunk(self.net(batch.flatten(1)), 2, dim=1)


class MLPDecoder(nn.Module):
    def __init__(self, exp_specs):
        super(MLPDecoder, self).__init__()

        self.img_size = exp_specs["img_size"]
        self.latent_dim = exp_specs["latent_dim"]
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, self.img_size ** 2),
            nn.Sigmoid(),
        )

    def forward(self, batch):
        out = self.net(batch)
        out = out.reshape(-1, 1, self.img_size, self.img_size)
        return out


class ConvEncoder(nn.Module):
    def __init__(self, exp_specs):
        super(ConvEncoder, self).__init__()
        self.latent_dim = exp_specs["latent_dim"]

        self.hidden_dims = [1, 16, 32, 64]
        self.net = []
        for in_channel, out_channel in zip(self.hidden_dims[:-1], self.hidden_dims[1:]):
            self.net += encode(in_channel, out_channel, bn=True)
        self.net += [
            nn.Conv2d(self.hidden_dims[-1], 2 * self.latent_dim, 3, 2, 1),
        ]

        self.image_to_latent = nn.Sequential(*self.net)

    def forward(self, batch):
        # return std and log_sigma respectively
        return torch.chunk(self.image_to_latent(batch), 2, dim=1)


class ConvDecoder(nn.Module):
    def __init__(self, exp_specs):
        super(ConvDecoder, self).__init__()
        self.latent_dim = exp_specs["latent_dim"]
        self.img_size = exp_specs["img_size"]

        self.net1 = nn.Sequential(
            nn.Linear(self.latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(32),
        )

        self.hidden_dims = [32, 64, 32, 8, 1]
        self.net2 = []
        for in_channel, out_channel in zip(
            self.hidden_dims[:-2], self.hidden_dims[1:-1]
        ):
            self.net2 += decode(in_channel, out_channel, bn=True)
        self.net2 += [
            nn.ConvTranspose2d(self.hidden_dims[-2], self.hidden_dims[-2], 4, 2, 1),
            nn.BatchNorm2d(self.hidden_dims[-2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.hidden_dims[-2], self.hidden_dims[-1], 3, 1, 1),
            nn.Tanh(),
        ]

        self.net2 = nn.Sequential(*self.net2)

    def forward(self, batch):
        # Map latent into appropriate size for transposed convolutions
        x = self.net1(batch)
        x = x.view(-1, 32, 1, 1)
        out = self.net2(x)
        out = out.view(-1, 1, self.img_size, self.img_size)
        return out
