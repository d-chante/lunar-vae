import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Wrap1d(nn.Module):
    def __init__(self, padding):
        super(Wrap1d, self).__init__()
        self.padding = padding

    def forward(self, x):
        x = torch.cat([x[:, :, -self.padding:], x,
                      x[:, :, :self.padding]], dim=2)
        return x


class VAE(nn.Module):

    def __init__(self, latent_variables):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            # 1e
            Wrap1d(padding=1),
            # 2e
            nn.ConstantPad1d(
                padding=3,
                value=0),
            # 3e
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # 4e
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # 5e
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # 6e
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # 7e
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # 8e
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # 9e
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # 10e
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU())

        # 11e
        self.fc_mu = nn.Conv1d(
            in_channels=32,
            out_channels=latent_variables,
            kernel_size=1,
            stride=1,
            padding=0)

        # 11e
        self.fc_logvar = nn.Conv1d(
            in_channels=32,
            out_channels=latent_variables,
            kernel_size=1,
            stride=1,
            padding=0)

        self.decoder = nn.Sequential(
            # 1d
            nn.ConvTranspose1d(
                in_channels=latent_variables,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # 2d
            nn.ConvTranspose1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # 3d
            nn.ConvTranspose1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # 4d
            nn.ConvTranspose1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # 5d
            nn.ConvTranspose1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # 6d
            nn.ConvTranspose1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # 7d
            nn.ConvTranspose1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # 8d
            nn.Conv1d(
                in_channels=32,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0),
            # 9d
            nn.ConstantPad1d(padding=-4, value=0))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta):
        reconstruction_loss = nn.functional.mse_loss(
            recon_x, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + beta * kl_divergence

    @staticmethod
    def save_state(
            epoch,
            model,
            optimizer,
            scheduler,
            training_loss,
            validation_loss,
            filepath):
        '''
        TBD
        '''
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'training_loss': training_loss,
            'validation_loss': validation_loss
        }
        torch.save(checkpoint, filepath)
        return f"Saved state to {filepath}"
