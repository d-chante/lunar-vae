#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchinfo import summary
from sklearn.preprocessing import StandardScaler

from lunar_vae import VAE, train, SampleLatentSpace, VisualizeLatentSpace
from utils import (
    GetConfig,
    CopyConfigFile,
    PrintAndLog,
    PrintConfig,
    LoadTemperatureDataV1,
    LoadTemperatureDataV2,
    GetDataMetrics,
    SplitAndNormalizeData,
    SetupOutputDir,
    PlotLosses,
    SaveLossesToCSV,
    GetMeanLatentValues
)


def main():
    # * * * * * * * * * * * * * * * *
    # ARG PARSER
    # * * * * * * * * * * * * * * * *
    parser = argparse.ArgumentParser(
        description="""
        Train, validate, and test Lunar VAE.

        Example usage:
            ./train_and_test.py -c /home/user/lunar-vae/cfg.yml -s
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        help="Path to configuration file",
        required=True)
    parser.add_argument(
        '-s',
        '--show',
        action='store_true',
        help='Show model summary and training/validation plot')
    parser.add_argument(
        '-v1',
        '--version-1',
        action='store_true',
        help='Use profile-v1 data (default: profile-v2)')

    args = parser.parse_args()

    # * * * * * * * * * * * * * * * *
    # PARAMETERS
    # * * * * * * * * * * * * * * * *
    meta_config, data_config, training_config = GetConfig(args.config)
    PrintConfig(meta_config, data_config, training_config)

    if meta_config is None:
        sys.exit(1)

    description = meta_config['description']
    input_data_path = data_config['input_data_path']
    output_dir = data_config['output_dir']
    device = torch.device(training_config['device'])
    latent_dim = int(training_config['latent_dim'])
    batch_size = int(training_config['batch_size'])
    epochs = int(training_config['epochs'])
    beta = float(training_config['beta'])
    learning_rate = float(training_config['learning_rate'])
    gamma = float(training_config['gamma'])

    input_dim = (1, 120)

    # * * * * * * * * * * * * * * * *
    # Output Dir Prep
    # * * * * * * * * * * * * * * * *
    model_dir = SetupOutputDir(output_dir, sub_folders=["images"])
    log_filepath = os.path.join(model_dir, 'training.log')

    # Copy the config file used to the model directory to help
    # track training parameters used
    CopyConfigFile(args.config, model_dir)

    # * * * * * * * * * * * * * * * *
    # Data Prep
    # * * * * * * * * * * * * * * * *
    # Load Data
    if args.v1:
        x_data = LoadTemperatureDataV1(input_data_path, device)
    else:
        x_data = LoadTemperatureDataV2(input_data_path, device)
        
    x_data_mean, x_data_std, x_data_min, x_data_max = GetDataMetrics(x_data)

    PrintAndLog(log_filepath, "Data Metrics:\n")
    PrintAndLog(log_filepath, f"\tMean: {x_data_mean:.4f}\n")
    PrintAndLog(log_filepath, f"\tSTD: {x_data_std:.4f}\n")
    PrintAndLog(log_filepath, f"\tMin Temp: {x_data_min:.4f}\n")
    PrintAndLog(log_filepath, f"\tMax Temp: {x_data_max:.4f}\n")

    # Normalize Data (Mean=0, STD=1)
    scaler = StandardScaler()
    train_loader, val_loader, test_loader = SplitAndNormalizeData(
        x_data, scaler, batch_size, device)

    PrintAndLog(log_filepath, "Datasets:\n")
    PrintAndLog(
        log_filepath,
        f"\tTraining Data: {len(train_loader.dataset)}\n")
    PrintAndLog(
        log_filepath,
        f"\tValidation Data: {len(val_loader.dataset)}\n")
    PrintAndLog(log_filepath, f"\tTest Data: {len(test_loader.dataset)}\n")

    # * * * * * * * * * * * * * * * *
    # VAE Setup
    # * * * * * * * * * * * * * * * *
    vae = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    if args.show:
        summary(vae, input_size=(batch_size, input_dim[0], input_dim[1]))

    # * * * * * * * * * * * * * * * *
    # Train and Validate
    # * * * * * * * * * * * * * * * *
    losses = train(
        vae, train_loader, val_loader,
        optimizer, scheduler, device, epochs,
        beta, model_dir)

    # Plot losses
    PlotLosses(losses, epochs, model_dir)

    # Save losses
    SaveLossesToCSV(losses, os.path.join(model_dir, "losses.csv"))

    PrintAndLog(log_filepath, "Best Validation Loss in Kelvins:\n")
    PrintAndLog(
        log_filepath,
        f"\t{min(losses['val_l1_losses'])*x_data_std:.4f} K\n")

    # * * * * * * * * * * * * * * * *
    # Sample & Reconstruct
    # * * * * * * * * * * * * * * * *
    z_sample = SampleLatentSpace(
        vae,
        test_loader,
        scaler,
        device,
        model_dir
    )

    # * * * * * * * * * * * * * * * *
    # Analyze Latent Space
    # * * * * * * * * * * * * * * * *
    mean_latent_vals = GetMeanLatentValues(z_sample, latent_dim)
    for i in range(latent_dim):
        print(f"z{i}: {mean_latent_vals[i]:.4f}")

    VisualizeLatentSpace(
        vae, latent_dim, z_sample,
        scaler, device, model_dir)


if __name__ == "__main__":
    main()
