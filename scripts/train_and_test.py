#!/usr/bin/env python3

import argparse
import datetime
import logging
import matplotlib.pyplot as plt
import os
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

        latent_variables = config['latent_variables']
        learning_rate = float(config['learning_rate'])
        patience = config['patience']
        factor = config['factor']
        beta = config['beta']
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
            level=logging.DEBUG,  # Set the base logging level
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
        logging.info(f"Patience: {patience}")
        logging.info(f"Factor: {factor}")
        logging.info(f"Beta: {beta}")
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
        model = VAE(latent_variables).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            verbose=True)

        if args.show:
            summary(model, input_dims, batch_size)
            time.sleep(1)  # Wait a moment to print summary

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

        logging.info(f"Training data: {len(train_data.dataset)}")
        logging.info(f"Validation data: {len(validation_data.dataset)}")
        logging.info(f"Test data: {len(test_data.dataset)}\n")

        # * * * * * * * * * * * * * * * *
        # TRAIN/VALIDATE
        # * * * * * * * * * * * * * * * *
        training_loss = []
        validation_loss = []
        epoch_time = []

        training_start_time = datetime.datetime.now()
        ms_path = os.path.join(
            output_dir,
            file_label,
            file_label +
            "_model_state.pt")

        logging.info("Training start\n")

        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch} of {num_epochs}")
            epoch_start_time = datetime.datetime.now()

            model.train()
            total_training_loss = 0
            for batch in train_data:
                batch = batch.to(device)
                optimizer.zero_grad()
                reconstructed, mu, logvar = model(batch)
                loss = VAE.loss_function(
                    reconstructed, batch, mu, logvar, beta)
                loss.backward()
                optimizer.step()
                total_training_loss += loss.item()

            avg_training_loss = total_training_loss / len(train_data.dataset)
            training_loss.append(avg_training_loss)
            logging.info(f"Training Loss: {avg_training_loss}")

            model.eval()
            total_validation_loss = 0
            with torch.no_grad():
                for batch in validation_data:
                    batch = batch.to(device)
                    reconstructed, mu, logvar = model(batch)
                    loss = VAE.loss_function(
                        reconstructed, batch, mu, logvar, beta)
                    total_validation_loss += loss.item()

            avg_validation_loss = total_validation_loss / \
                len(validation_data.dataset)
            validation_loss.append(avg_validation_loss)
            logging.info(f"Validation Loss: {avg_validation_loss}")

            elapsed_time = ut.ElapsedSecondsSince(epoch_start_time)
            epoch_time.append(elapsed_time)

            logging.info(f"Elapsed time: {ut.FormatSeconds(elapsed_time)}")

            logging.info(
                model.save_state(
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    avg_training_loss,
                    avg_validation_loss,
                    ms_path))

            scheduler.step(avg_validation_loss)

            logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}\n")

        logging.info(
            model.save_state(
                epoch,
                model,
                optimizer,
                scheduler,
                avg_training_loss,
                avg_validation_loss,
                ms_path))

        avg_epoch_time = ut.FormatSeconds(sum(epoch_time) / num_epochs)
        total_training_time = ut.FormatSeconds(
            ut.ElapsedSecondsSince(training_start_time))

        logging.info(f"Average epoch time: {avg_epoch_time}")
        logging.info(f"Total training time: {total_training_time}\n")

        tvl_path = os.path.join(
            output_dir,
            file_label,
            file_label +
            "_loss.csv")
        ut.SaveLoss2Csv(training_loss, validation_loss, tvl_path)
        logging.info(f"Saved training and validation loss to {tvl_path}")

        plt.figure(figsize=(10, 5))
        plt.plot(training_loss, label='Training Loss')
        plt.plot(validation_loss, label='Validation Loss')
        plt.title('Training/Validation Loss per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.ioff()

        tvplot_path = os.path.join(output_dir, file_label, file_label + ".png")
        plt.savefig(tvplot_path)
        logging.info(f"Saved training and validation plot {tvplot_path}\n")

        if args.show:
            plt.show()

        plt.close()

        # * * * * * * * * * * * * * * * *
        # TEST
        # * * * * * * * * * * * * * * * *
        logging.info("Testing start")
        test_loss = 0

        latent_variables_mu = []
        latent_variables_logvar = []

        model.eval()
        with torch.no_grad():
            for batch in test_data:
                batch = batch.to(device)
                reconstructed, mu, logvar = model(batch)
                latent_variables_mu.append(mu.cpu().numpy())
                latent_variables_logvar.append(logvar.cpu().numpy())
                loss = VAE.loss_function(
                    reconstructed, batch, mu, logvar, beta)
                test_loss = loss.item()

        avg_test_loss = test_loss / len(test_data.dataset)
        logging.info(f"Test Loss: {avg_test_loss}\n")

        m_path = os.path.join(
            output_dir,
            file_label,
            file_label +
            "_overall_metrics.txt")
        ut.SaveOtherMetrics(
            [len(train_data.dataset), len(validation_data.dataset), len(test_data.dataset)],
            metrics[0],
            metrics[1],
            optimizer.param_groups[0]['lr'],
            avg_epoch_time,
            total_training_time,
            avg_test_loss,
            m_path)
        logging.info(f"Saved other metrics to {m_path}")

        # * * * * * * * * * * * * * * * *
        # LATENT VARIABLES
        # * * * * * * * * * * * * * * * *
        lvmu_path = os.path.join(
            output_dir,
            file_label,
            file_label +
            "_latent_variables_mu.csv")
        ut.SaveLatentVariables2Csv(latent_variables_mu, lvmu_path)
        logging.info(f"Saved latent variables mean to {lvmu_path}")

        lvlogvar_path = os.path.join(
            output_dir,
            file_label,
            file_label +
            "_latent_variables_logvar.csv")
        ut.SaveLatentVariables2Csv(latent_variables_mu, lvlogvar_path)
        logging.info(f"Saved latent variables logvar to {lvlogvar_path}")


if __name__ == "__main__":
    main()
