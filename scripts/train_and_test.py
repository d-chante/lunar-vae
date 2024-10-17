#!/usr/bin/env python3

import argparse
import datetime
import logging
import matplotlib.pyplot as plt
import os
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torchsummary import summary

from lunar_vae import VAE
from utils import Utils


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
        '-i',
        '--info',
        action='store_true',
        help='Show batch info (note: results in large log files)'
    )

    try:
        args = parser.parse_args()
    except SystemExit as e:
        if e.code != 0:
            raise
        else:
            args = None

    if args:

        # * * * * * * * * * * * * * * * *
        # PARAMETERS
        # * * * * * * * * * * * * * * * *
        ut = Utils()

        dirs, config = ut.GetConfig(args.config)

        profiles_dir = dirs['profiles_directory']
        output_dir = dirs['output_directory']
        results_filepath = dirs['results_filepath']

        latent_variables = config['latent_variables']
        learning_rate = float(config['learning_rate'])
        gamma = float(config['gamma'])
        dropout = config['dropout']
        num_epochs = config['epochs']
        batch_size = config['batch_size']
        gpu = config['gpu']

        input_dims = (1, 120)

        # * * * * * * * * * * * * * * * *
        # LABEL
        # * * * * * * * * * * * * * * * *
        file_label = ut.GenerateFilename()
        ut.CreateDirectory(os.path.join(output_dir, file_label))

        # * * * * * * * * * * * * * * * *
        # LOGGER
        # * * * * * * * * * * * * * * * *
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    os.path.join(
                        output_dir,
                        file_label,
                        file_label +
                        ".log")),
                logging.StreamHandler()
            ])

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)

        logging.info(f"Latent Variables: {latent_variables}")
        logging.info(f"Learning Rate: {learning_rate}")
        logging.info(f"Gamma: {gamma}")
        logging.info(f"Beta: 0.2")
        logging.info(f"Dropout: {dropout}")
        logging.info(f"Number of Epochs: {num_epochs}")
        logging.info(f"Batch Size: {batch_size}")
        logging.info(f"GPU: {gpu}\n")

        # * * * * * * * * * * * * * * * *
        # DEVICE
        # * * * * * * * * * * * * * * * *
        device = torch.device(
            f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        logging.info(f"Training with: {device}")

        # * * * * * * * * * * * * * * * *
        # MODEL
        # * * * * * * * * * * * * * * * *
        model = VAE(latent_variables, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=gamma)

        if args.show:
            summary(model, input_dims)

        # * * * * * * * * * * * * * * * *
        # DATA
        # * * * * * * * * * * * * * * * *
        logging.info(f"Loading data...")
        data, metrics = ut.LoadData(profiles_dir, batch_size)

        train_data = data[0]
        validation_data = data[1]
        test_data = data[2]

        logging.info(f"Data Mean: {metrics[0]}")
        logging.info(f"Data Standard Deviation: {metrics[1]}\n")

        logging.info(
            f"Training data: {len(train_data.dataset)} profiles in {len(train_data)} batches")
        logging.info(
            f"Validation data: {len(validation_data.dataset)} profiles in {len(validation_data)} batches")
        logging.info(
            f"Test data: {len(test_data.dataset)} profiles in {len(test_data)} batches\n")

        # * * * * * * * * * * * * * * * *
        # TRAIN/VALIDATE
        # * * * * * * * * * * * * * * * *
        training_reconstruction_loss = []
        training_kl_divergence = []
        training_elbo_loss = []

        validation_reconstruction_loss = []
        validation_kl_divergence = []
        validation_elbo_loss = []

        validation_mu = []
        validation_logvar = []

        epoch_time = []

        training_start_time = datetime.datetime.now()
        ms_path = os.path.join(
            output_dir,
            file_label,
            file_label +
            "_model_state.pt")

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            logging.info(
                f"------------------Epoch {epoch+1} of {num_epochs}------------------")
            epoch_start_time = datetime.datetime.now()
            beta = 0.2

            model.train()
            logging.info("Training...")

            epoch_reconstruction_loss = 0
            epoch_kl_divergence = 0
            epoch_elbo_loss = 0

            for batch_num, batch in enumerate(train_data, 1):
                # Predict
                batch = batch.to(device)
                reconstructed, mu, logvar = model(batch)

                # Loss Calculation
                reconstruction_loss = VAE.reconstruction_loss(
                    reconstructed, batch)
                kl_divergence = VAE.kl_divergence(logvar, mu)
                elbo_loss = VAE.elbo_loss(
                    reconstruction_loss, kl_divergence, beta)

                # Back Propagation
                optimizer.zero_grad()
                elbo_loss.backward()
                optimizer.step()

                # Track Losses over batches
                epoch_reconstruction_loss += reconstruction_loss.item()
                epoch_kl_divergence += kl_divergence.item()
                epoch_elbo_loss += elbo_loss.item()

                if args.info:
                    logging.info(
                        f"Batch {batch_num} Training Losses\n\tReconstruction Loss: {reconstruction_loss.item()}\n\tKL Divergence: {kl_divergence.item()}\n\tELBO Loss: {elbo_loss.item()}\n")

            # Get average of loss per batch
            avg_reconstruction_loss = epoch_reconstruction_loss / \
                len(train_data.dataset)
            training_reconstruction_loss.append(avg_reconstruction_loss)

            avg_kl_divergence = epoch_kl_divergence / len(train_data.dataset)
            training_kl_divergence.append(avg_kl_divergence)

            avg_training_elbo_loss = epoch_elbo_loss / len(train_data.dataset)
            training_elbo_loss.append(avg_training_elbo_loss)

            logging.info(
                f"Epoch {epoch+1} Training Losses\n\tReconstruction Loss: {avg_reconstruction_loss}\n\tKL Divergence: {avg_kl_divergence}\n\tELBO Loss: {avg_training_elbo_loss}\n")

            model.eval()
            logging.info("Validate...")

            epoch_reconstruction_loss = 0
            epoch_kl_divergence = 0
            epoch_elbo_loss = 0

            with torch.no_grad():
                for batch_num, batch in enumerate(validation_data, 1):
                    # Predict
                    batch = batch.to(device)
                    reconstructed, mu, logvar = model(batch)

                    # Loss Calculation
                    reconstruction_loss = VAE.reconstruction_loss(
                        reconstructed, batch)
                    kl_divergence = VAE.kl_divergence(logvar, mu)
                    elbo_loss = VAE.elbo_loss(
                        reconstruction_loss, kl_divergence, beta)

                    # Track Losses over batches
                    epoch_reconstruction_loss += reconstruction_loss.item()
                    epoch_kl_divergence += kl_divergence.item()
                    epoch_elbo_loss += elbo_loss.item()

                    if args.info:
                        logging.info(
                            f"Batch {batch_num} Validation Losses\n\tReconstruction Loss: {reconstruction_loss.item()}\n\tKL Divergence: {kl_divergence.item()}\n\tELBO Loss: {elbo_loss.item()}\n")

                    # Store mu and logvar as np arrays
                    validation_mu.append(mu.cpu().numpy())
                    validation_logvar.append(logvar.cpu().numpy())

            avg_reconstruction_loss = epoch_reconstruction_loss / \
                len(validation_data.dataset)
            validation_reconstruction_loss.append(avg_reconstruction_loss)

            avg_kl_divergence = epoch_kl_divergence / len(validation_data.dataset)
            validation_kl_divergence.append(avg_kl_divergence)

            avg_validation_elbo_loss = epoch_elbo_loss / len(validation_data.dataset)
            validation_elbo_loss.append(avg_validation_elbo_loss)

            logging.info(
                f"Epoch {epoch+1} Validation Losses\n\tReconstruction Loss: {avg_reconstruction_loss}\n\tKL Divergence: {avg_kl_divergence}\n\tELBO Loss: {avg_validation_elbo_loss}\n")

            elapsed_time = ut.ElapsedSecondsSince(epoch_start_time)
            epoch_time.append(elapsed_time)

            scheduler.step()

            logging.info(f"Elapsed time: {ut.FormatSeconds(elapsed_time)}")

            if avg_validation_elbo_loss < best_val_loss:
                best_val_loss = avg_validation_elbo_loss
                logging.info(
                    f"Saving model with best validation ELBO loss {best_val_loss}")
                logging.info(
                    model.save_state(
                        epoch + 1,
                        model,
                        optimizer,
                        scheduler,
                        avg_training_elbo_loss,
                        avg_validation_elbo_loss,
                        ms_path))

            logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}\n")

        avg_epoch_time = ut.FormatSeconds(sum(epoch_time) / num_epochs)
        total_training_time = ut.FormatSeconds(
            ut.ElapsedSecondsSince(training_start_time))

        logging.info(f"Best validation ELBO Loss: {best_val_loss}")
        logging.info(f"Average epoch time: {avg_epoch_time}")
        logging.info(f"Total training time: {total_training_time}\n")

        # * * * * * * * * * * * * * * * *
        # SAVE LOSSES
        # * * * * * * * * * * * * * * * *
        tvl_path = os.path.join(
            output_dir,
            file_label,
            file_label +
            "_loss.csv")
        ut.SaveLoss2Csv(
            training_reconstruction_loss,
            training_kl_divergence,
            training_elbo_loss,
            validation_reconstruction_loss,
            validation_kl_divergence,
            validation_elbo_loss,
            tvl_path)
        logging.info(f"Saved training and validation loss to {tvl_path}")

        # * * * * * * * * * * * * * * * *
        # SAVE LATENT VARS
        # * * * * * * * * * * * * * * * *
        lvmu_path = os.path.join(
            output_dir,
            file_label,
            file_label +
            "_latent_variables_mu.csv")
        ut.SaveLatentVariables2Csv(validation_mu, lvmu_path)
        logging.info(f"Saved latent variables mean to {lvmu_path}")

        lvlogvar_path = os.path.join(
            output_dir,
            file_label,
            file_label +
            "_latent_variables_logvar.csv")
        ut.SaveLatentVariables2Csv(validation_logvar, lvlogvar_path)
        logging.info(f"Saved latent variables logvar to {lvlogvar_path}")

        # * * * * * * * * * * * * * * * *
        # PLOT LOSSES
        # * * * * * * * * * * * * * * * *
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

        axes[0].plot(training_elbo_loss, label="Training ELBO Loss")
        axes[0].plot(validation_elbo_loss, label="Validation ELBO Loss")
        axes[0].set_title("Training/Validation ELBO Loss per Epoch")
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('ELBO Loss')
        axes[0].legend()

        axes[1].plot(
            training_reconstruction_loss,
            label='Training Reconstruction Loss')
        axes[1].plot(
            validation_reconstruction_loss,
            label='Validation Reconstruction Loss')
        axes[1].set_title('Training/Validation Reconstruction Loss per Epoch')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Reconstruction Loss')
        axes[1].legend()

        axes[2].plot(training_kl_divergence, label='Training KL Divergence')
        axes[2].plot(
            validation_kl_divergence,
            label='Validation KL Divergence')
        axes[2].set_title('Training/Validation KL Divergence per Epoch')
        axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel('KL Divergence')
        axes[2].legend()

        plt.tight_layout()

        plot_path = os.path.join(output_dir, file_label, file_label + ".png")
        plt.savefig(plot_path)
        logging.info(f"Saved figure to {plot_path}\n")

        if args.show:
            plt.show()

        plt.close()

        # * * * * * * * * * * * * * * * *
        # TEST
        # * * * * * * * * * * * * * * * *
        logging.info("Testing start")
        test_reconstruction_loss = 0
        test_kl_divergence = 0
        test_elbo_loss = 0

        model.eval()
        with torch.no_grad():
            for batch_num, batch in enumerate(test_data, 1):
                # Predict
                batch = batch.to(device)
                reconstructed, mu, logvar = model(batch)

                # Calculate Losses
                reconstruction_loss = VAE.reconstruction_loss(
                    reconstructed, batch)
                kl_divergence = VAE.kl_divergence(logvar, mu)
                elbo_loss = VAE.elbo_loss(
                    reconstruction_loss, kl_divergence, beta)

                # Track losses
                test_reconstruction_loss += reconstruction_loss.item()
                test_kl_divergence += kl_divergence.item()
                test_elbo_loss += elbo_loss.item()

                if args.info:
                    logging.info(
                        f"Batch {batch_num} Test Losses\n\tReconstruction Loss: {reconstruction_loss.item()}\n\tKL Divergence: {kl_divergence.item()}\n\tELBO Loss: {elbo_loss.item()}\n")

        avg_test_reconstruction_loss = test_reconstruction_loss / \
            len(test_data.dataset)
        avg_test_kl_divergence = test_kl_divergence / len(test_data.dataset)
        avg_test_elbo_loss = test_elbo_loss / len(test_data.dataset)

        logging.info(
            f"Test Losses\n\tReconstruction Loss: {avg_test_reconstruction_loss}\n\tKL Divergence: {avg_test_kl_divergence}\n\tELBO Loss: {avg_test_elbo_loss}\n")

        # * * * * * * * * * * * * * * * *
        # SAVE METRICS
        # * * * * * * * * * * * * * * * *
        ut.SaveMetrics(
            file_label,
            args.config,
            latent_variables,
            learning_rate,
            gamma,
            beta,
            dropout,
            num_epochs,
            batch_size,
            gpu,
            [len(train_data.dataset), len(validation_data.dataset), len(test_data.dataset)],
            metrics[0],
            metrics[1],
            optimizer.param_groups[0]['lr'],
            avg_epoch_time,
            total_training_time,
            avg_training_elbo_loss,
            avg_validation_elbo_loss,
            avg_test_elbo_loss,
            results_filepath
        )
        logging.info(f"Saved other metrics to {results_filepath}")


if __name__ == "__main__":
    main()
