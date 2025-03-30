
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


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


class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.wrap_1e = Wrap1d(wrap_size=1)
        self.pad_2e = nn.ConstantPad1d(padding=3, value=0)

        self.conv_3e = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn_3e = nn.BatchNorm1d(32)

        self.conv_4e = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=2,
            stride=2,
            padding=0)
        self.bn_4e = nn.BatchNorm1d(32)

        self.conv_5e = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=2,
            stride=2,
            padding=0)
        self.bn_5e = nn.BatchNorm1d(32)

        self.conv_6e = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=2,
            stride=2,
            padding=0)
        self.bn_6e = nn.BatchNorm1d(32)

        self.conv_7e = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=2,
            stride=2,
            padding=0)
        self.bn_7e = nn.BatchNorm1d(32)

        self.conv_8e = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=2,
            stride=2,
            padding=0)
        self.bn_8e = nn.BatchNorm1d(32)

        self.conv_9e = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=2,
            stride=2,
            padding=0)
        self.bn_9e = nn.BatchNorm1d(32)

        self.conv_10e = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=2,
            stride=2,
            padding=0)
        self.bn_10e = nn.BatchNorm1d(32)

        self.z_mean = nn.Conv1d(
            32,
            latent_dim,
            kernel_size=1,
            stride=1,
            padding=0)
        self.z_log_var = nn.Conv1d(
            32, latent_dim, kernel_size=1, stride=1, padding=0)
        self.sampling = Sampling()

    def forward(self, x):
        x = self.wrap_1e(x)
        x = self.pad_2e(x)
        x = torch.relu(self.bn_3e(self.conv_3e(x)))
        x = torch.relu(self.bn_4e(self.conv_4e(x)))
        x = torch.relu(self.bn_5e(self.conv_5e(x)))
        x = torch.relu(self.bn_6e(self.conv_6e(x)))
        x = torch.relu(self.bn_7e(self.conv_7e(x)))
        x = torch.relu(self.bn_8e(self.conv_8e(x)))
        x = torch.relu(self.bn_9e(self.conv_9e(x)))
        x = torch.relu(self.bn_10e(self.conv_10e(x)))

        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z, z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.convt_1d = nn.ConvTranspose1d(
            in_channels=latent_dim,
            out_channels=32,
            kernel_size=2,
            stride=2,
            padding=0)
        self.bn_1d = nn.BatchNorm1d(32)

        self.convt_2d = nn.ConvTranspose1d(
            in_channels=32,
            out_channels=32,
            kernel_size=2,
            stride=2,
            padding=0)
        self.bn_2d = nn.BatchNorm1d(32)

        self.convt_3d = nn.ConvTranspose1d(
            in_channels=32,
            out_channels=32,
            kernel_size=2,
            stride=2,
            padding=0)
        self.bn_3d = nn.BatchNorm1d(32)

        self.convt_4d = nn.ConvTranspose1d(
            in_channels=32,
            out_channels=32,
            kernel_size=2,
            stride=2,
            padding=0)
        self.bn_4d = nn.BatchNorm1d(32)

        self.convt_5d = nn.ConvTranspose1d(
            in_channels=32,
            out_channels=32,
            kernel_size=2,
            stride=2,
            padding=0)
        self.bn_5d = nn.BatchNorm1d(32)

        self.convt_6d = nn.ConvTranspose1d(
            in_channels=32,
            out_channels=32,
            kernel_size=2,
            stride=2,
            padding=0)
        self.bn_6d = nn.BatchNorm1d(32)

        self.convt_7d = nn.ConvTranspose1d(
            in_channels=32,
            out_channels=32,
            kernel_size=2,
            stride=2,
            padding=0)
        self.bn_7d = nn.BatchNorm1d(32)

        self.convt_8d = nn.Conv1d(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0)

        self.crop_9d = Crop1d(crop_size=4)

    def forward(self, z):
        x = torch.relu(self.bn_1d(self.convt_1d(z)))
        x = torch.relu(self.bn_2d(self.convt_2d(x)))
        x = torch.relu(self.bn_3d(self.convt_3d(x)))
        x = torch.relu(self.bn_4d(self.convt_4d(x)))
        x = torch.relu(self.bn_5d(self.convt_5d(x)))
        x = torch.relu(self.bn_6d(self.convt_6d(x)))
        x = torch.relu(self.bn_7d(self.convt_7d(x)))
        x = torch.relu(self.convt_8d(x))
        x = self.crop_9d(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x, inference=False):
        z, z_mean, z_log_var = self.encoder(x)
        if inference:
            x_recon = self.decoder(z_mean)
        else:
            x_recon = self.decoder(z)
        return x_recon, z_mean, z_log_var


def calculate_loss(x_recon, x, z_mean, z_log_var, beta=0.2):
    l2_recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
    l1_recon_loss = F.l1_loss(x_recon, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    elbo_loss = l2_recon_loss + beta * kl_div
    return l1_recon_loss, l2_recon_loss, kl_div, elbo_loss


def train(
        vae,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        epochs,
        beta,
        model_dir):
    '''
    @brief Performs training/validation
    '''
    # Ensure VAE is explicitly moved to the target device
    vae.to(device)

    # Track losses over epochs
    train_l1_losses = []
    train_l2_losses = []
    train_kl_losses = []
    val_l1_losses = []
    val_l2_losses = []
    val_kl_losses = []
    lr_history = []

    best_val_loss = float('inf')

    # Training Loop
    for epoch in range(epochs):
        # * * * * * * * * * * *
        # Training
        # * * * * * * * * * * *
        vae.train()
        epoch_train_l1_loss = 0
        epoch_train_l2_loss = 0
        epoch_train_kl_loss = 0
        best_train_elbo_loss = float('inf')

        progress_bar = tqdm(
            train_loader,
            desc=f"[Training] Epoch {epoch+1}/{epochs}")

        for x, _ in progress_bar:
            x = x.to(device)

            optimizer.zero_grad()
            x_recon, z_mean, z_log_var = vae(x)

            # Get all losses from the loss function
            l1_loss, l2_loss, kl_loss, elbo_loss = calculate_loss(
                x_recon, x, z_mean, z_log_var, beta)

            # Backpropagation and optimization
            elbo_loss.backward()
            optimizer.step()

            # Accumulate losses for tracking
            epoch_train_l1_loss += l1_loss.item()
            epoch_train_l2_loss += l2_loss.item()
            epoch_train_kl_loss += kl_loss.item()

            if elbo_loss.item() < best_train_elbo_loss:
                best_train_elbo_loss = elbo_loss.item()

            progress_bar.set_postfix(best_loss=f"{best_train_elbo_loss:.4f}")

        # Log current learning rate
        current_lr = scheduler.get_last_lr()[0]
        lr_history.append(current_lr)

        # Adjust the learning rate
        scheduler.step()

        # Calculate average losses for the epoch
        epoch_avg_l1_loss = epoch_train_l1_loss / len(train_loader)
        epoch_avg_l2_loss = epoch_train_l2_loss / len(train_loader)
        epoch_avg_kl_loss = epoch_train_kl_loss / len(train_loader)

        train_l1_losses.append(epoch_avg_l1_loss)
        train_l2_losses.append(epoch_avg_l2_loss)
        train_kl_losses.append(epoch_avg_kl_loss)

        print(f"[Training] Epoch {epoch+1}/{epochs}, L1 Loss: {epoch_avg_l1_loss:.4f}, L2 Loss: {epoch_avg_l2_loss:.4f}, KL Loss: {epoch_avg_kl_loss:.4f}, LR: {current_lr:.6f}")

        # * * * * * * * * * * *
        # Validation
        # * * * * * * * * * * *
        vae.eval()
        epoch_val_l1_loss = 0
        epoch_val_l2_loss = 0
        epoch_val_kl_loss = 0

        progress_bar = tqdm(
            val_loader,
            desc=f"[Validation] Epoch {epoch+1}/{epochs}")

        with torch.no_grad():
            for x, _ in progress_bar:
                x = x.to(device)

                x_recon, z_mean, z_log_var = vae(x, inference=True)

                # Get validation losses
                l1_loss, l2_loss, kl_loss, _ = calculate_loss(
                    x_recon, x, z_mean, z_log_var, beta)

                # Accumulate losses for tracking
                epoch_val_l1_loss += l1_loss.item()
                epoch_val_l2_loss += l2_loss.item()
                epoch_val_kl_loss += kl_loss.item()

        # Calculate average validation losses
        avg_val_l1_loss = epoch_val_l1_loss / len(val_loader)
        avg_val_l2_loss = epoch_val_l2_loss / len(val_loader)
        avg_val_kl_loss = epoch_val_kl_loss / len(val_loader)

        val_l1_losses.append(avg_val_l1_loss)
        val_l2_losses.append(avg_val_l2_loss)
        val_kl_losses.append(avg_val_kl_loss)

        print(
            f"[Validation] Epoch {epoch+1}/{epochs}, Val L1 Loss: {avg_val_l1_loss:.4f}, Val L2 Loss: {avg_val_l2_loss:.4f}, Val KL Loss: {avg_val_kl_loss:.4f}")

        if avg_val_l1_loss < best_val_loss:
            best_val_loss = avg_val_l1_loss
            SaveModel(
                vae,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                model_dir)
            print(f"[Validation] Saved checkpoint in '{model_dir}'")

    return {
        "train_l1_losses": train_l1_losses,
        "train_l2_losses": train_l2_losses,
        "train_kl_losses": train_kl_losses,
        "val_l1_losses": val_l1_losses,
        "val_l2_losses": val_l2_losses,
        "val_kl_losses": val_kl_losses,
        "lr_history": lr_history
    }


def SampleLatentSpace(
        vae, data_loader, scaler, device, model_dir):
    '''
    @brief Samples the latent space and generates traversal plots.
    '''
    vae.eval()

    x_fit = np.linspace(0, 24, 120)

    z_sample = []
    num_samples = math.floor(len(data_loader.dataset) * 0.10)
    plot_sample_freq = math.floor(num_samples * 0.01)
    sample_count = 0

    for batch_idx, (x_batch, _) in tqdm(
            enumerate(data_loader), total=math.ceil(
            num_samples / data_loader.batch_size), desc="[Sampling]"):
        if sample_count >= num_samples:
            break

        x_batch = x_batch.to(device)

        for i in range(x_batch.size(0)):
            if sample_count >= num_samples:
                break

            xx = x_batch[i].unsqueeze(0)

            with torch.no_grad():
                xx_cpu = xx.squeeze(0).cpu().numpy().reshape(
                    1, -1)  # Adjust dimensions for scaler
                xx_denorm = scaler.inverse_transform(xx_cpu).squeeze()

            with torch.no_grad():
                x_recon, z_mean, z_log_var = vae(xx, inference=True)
                x_recon = x_recon.squeeze(0).cpu().numpy().reshape(1, -1)
                x_recon_denorm = scaler.inverse_transform(x_recon).squeeze()

                z_sample.append(z_mean.squeeze().cpu().numpy())

            # Plot % of samples
            if sample_count % plot_sample_freq == 0:
                fig = plt.figure(figsize=(5, 5))
                plt.plot(x_fit, xx_denorm, label="Original")
                plt.plot(x_fit, x_recon_denorm)
                plt.legend()
                plt.ylim(25, 425)
                plt.xlim(0, 24)
                plt.xticks([0, 6, 12, 18, 24])
                plt.xlabel("Local lunar time (hours)")
                plt.ylabel("Temperature (K)")
                plt.title(f"Sample {sample_count+1}")
                fig.savefig(f"{model_dir}/reconstruction_images/sample_{sample_count}.png")
                plt.close()
                
            sample_count += 1
            
    return np.array(z_sample)


def VisualizeLatentSpace(
        vae,
        latent_dim,
        z_arrays,
        scaler,
        device,
        model_dir,
        num_traversals=11):
    vae.eval()
    x_fit = np.linspace(0, 24, 120)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()

    for i in range(latent_dim):
        ax = axs[i]

        vals = np.linspace(np.min(z_arrays[:, i]), np.max(
            z_arrays[:, i]), num_traversals)

        for val in tqdm(vals, desc=f"Traversing Latent Dimension z{i}"):
            with torch.no_grad():
                # Create a latent vector with one dimension varied
                z = torch.zeros((1, latent_dim), device=device)
                z[0, i] = val

                prediction = vae.decoder(z.unsqueeze(2))
                prediction = prediction.squeeze(0).cpu().numpy().reshape(1, -1)
                prediction_denorm = scaler.inverse_transform(
                    prediction).squeeze()

            ax.plot(x_fit, prediction_denorm)

        # Set subplot title and labels
        ax.set_title(f"Latent Variable {i+1}")
        ax.set_ylim(25, 425)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xlabel("Local lunar time (hours)")
        ax.set_ylabel("Temperature (K)")

    plt.tight_layout()
    fig.savefig(f"{model_dir}/latent_space_visualization.png")
    plt.show()
    plt.close()


def SaveModel(vae, optimizer, scheduler, epoch, val_loss, model_dir):
    '''
    @brief Saves model
    '''
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss
    }
    torch.save(checkpoint, os.path.join(model_dir, 'vae_model.pth'))


def LoadModel(vae, optimizer, scheduler, model_path, device):
    '''
    @brief Loads model, including optimizer and scheduler states
    '''
    checkpoint = torch.load(model_path, map_location=device)

    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.to(device)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', -1)
    val_loss = checkpoint.get('val_loss', float('inf'))

    return epoch, val_loss
