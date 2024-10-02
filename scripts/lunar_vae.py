import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Wrap1d(nn.Module):
    def __init__(self, wrap_size=1):
        super(Wrap1d, self).__init__()
        self.wrap_size = wrap_size

    def forward(self, x):
        wrapped_part_start = x[:, :, -self.wrap_size:]  
        wrapped_part_end = x[:, :, :self.wrap_size]  
        x_wrapped = torch.cat([wrapped_part_start, x, wrapped_part_end], dim=2) 
        return x_wrapped
    
class Crop1d(nn.Module):
    def __init__(self, crop_size):
        super(Crop1d, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        start = self.crop_size
        end = -self.crop_size if self.crop_size != 0 else None
        return x[:, :, start:end]

class VAE(nn.Module):

    def __init__(self, latent_variables, dropout):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            # 1e
            Wrap1d(wrap_size=1),
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
            nn.Dropout(p=dropout),
            # 4e
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # 5e
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # 6e
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # 7e
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # 8e
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # 9e
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # 10e
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout))

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
            nn.Dropout(p=dropout),
            # 2d
            nn.ConvTranspose1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # 3d
            nn.ConvTranspose1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # 4d
            nn.ConvTranspose1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # 5d
            nn.ConvTranspose1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # 6d
            nn.ConvTranspose1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # 7d
            nn.ConvTranspose1d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # 8d
            nn.Conv1d(
                in_channels=32,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0),
            # 9d
            Crop1d(crop_size=4))

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
    def reconstruction_loss(recon_x, x):
        return nn.functional.l1_loss(recon_x, x, reduction='mean')

    @staticmethod
    def kl_divergence(logvar, mu):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    @staticmethod
    def elbo_loss(reconstruction_loss, kl_divergence, beta):
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
    
    @staticmethod
    def load_state(state_path, latent_variables=4, dropout=0, device='cpu'):
        '''
        @brief Loads and returns a VAE model state
        @param state_path Filepath to the state model dictionary
        @param latent_variables Number of latent variables
        @param dropout Dropout layer percentage (0.0-1.0)
        @param device Target CPU or GPU
        '''
        checkpoint = torch.load(state_path, map_location=device)
        model = VAE(latent_variables, dropout)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model