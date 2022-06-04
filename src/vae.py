import torch
from torch import nn

from src.models import *


class VAE(nn.Module):
    def __init__(self, exp_specs):
        super(VAE, self).__init__()

        self.latent_dim = exp_specs["latent_dim"]
        self.img_size = exp_specs["img_size"]
        self.arch = exp_specs["arch"]
        if self.arch == "conv":
            self.encoder = ConvEncoder(exp_specs)
            self.decoder = ConvDecoder(exp_specs)
        elif self.arch == "mlp":
            self.encoder = MLPEncoder(exp_specs)
            self.decoder = MLPDecoder(exp_specs)

    def forward(self, batch):
        if self.arch == "conv":
            mu, log_sigma = self.encoder(batch)
            mu, log_sigma = (
                mu.reshape(-1, self.latent_dim),
                log_sigma.reshape(-1, self.latent_dim),
            )
            #  Sample from N(mu, sigma)
            std = torch.exp(0.5 * log_sigma)
            eps = torch.randn_like(std)
            recons = self.decoder(eps * std + mu)
        elif self.arch == "mlp":
            mu, log_sigma = self.encoder(batch)
            #  Sample from N(mu, sigma)
            std = torch.exp(0.5 * log_sigma)
            eps = torch.randn_like(std)
            recons = self.decoder(eps * std + mu)

        return mu, log_sigma, recons

    def generate(self, latents):
        """ 
        latents: [B * latent_dim]
        """
        return self.decoder(latents).reshape(-1, 1, self.img_size, self.img_size)
