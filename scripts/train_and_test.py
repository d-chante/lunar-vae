import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import torch
from torchsummary import summary

from lunar_vae import VAE
from utils import Utils


def main():

    if len(sys.argv) < 2:
        print("CFG_FILEPATH required")
        sys.exit(1)

    ut = Utils()

    # * * * * * * * * * * * * * * * *
    # PARAMETERS
    # * * * * * * * * * * * * * * * *
    dirs, config = ut.GetConfig(sys.argv[1])

    profiles_dir = dirs['profiles_directory']
    output_dir = dirs['output_directory']

    latent_variables = config['latent_variables']
    learning_rate = float(config['learning_rate'])
    beta = config['beta']
    num_epochs = config['epochs']
    batch_size = config['batch_size']
    n_splits = config['n_splits']
    gpu = config['gpu']
    input_dims = (1, 120)

    # * * * * * * * * * * * * * * * *
    # LOGGER
    # * * * * * * * * * * * * * * * *
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(stream_handler)

    # * * * * * * * * * * * * * * * *
    # CUDA
    # * * * * * * * * * * * * * * * *
    if torch.cuda.is_available():
        logging.info(f"CUDA available - default device: {torch.cuda.current_device()}\n")
    else:
        logging.info(f"CUDA not available - training on CPU\n")

    # * * * * * * * * * * * * * * * *
    # MODEL
    # * * * * * * * * * * * * * * * *
    model = VAE(latent_variables)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    summary(model, input_dims, batch_size)
    time.sleep(1)  # Wait a moment to print summary

    # * * * * * * * * * * * * * * * *
    # DATA
    # * * * * * * * * * * * * * * * *
    data, metrics = ut.LoadData(profiles_dir, batch_size)

    train_data = data[0]
    validation_data = data[1]
    test_data = data[2]

    logging.info(f"Data Mean: {metrics[0]}")
    logging.info(f"Data Standard Deviation: {metrics[1]}\n")

    logging.info(f"Training data: {len(train_data.dataset)}")
    logging.info(f"Validation data: {len(validation_data.dataset)}")
    logging.info(f"Test data: {len(test_data.dataset)}\n")

    label = ut.GenerateFilename()

    # * * * * * * * * * * * * * * * *
    # TRAIN/VALIDATE
    # * * * * * * * * * * * * * * * *
    training_loss = []
    validation_loss = []

    epoch_time = []

    training_start_time = datetime.datetime.now()
    ms_path = os.path.join(output_dir, label + "_model_state.pt")

    logging.info("Training\n")

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch} of {num_epochs}")
        epoch_start_time = datetime.datetime.now()

        model.train()
        total_training_loss = 0
        for batch in train_data:
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(batch)
            loss = VAE.loss_function(reconstructed, batch, mu, logvar, beta)
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

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_training_loss,
        }, ms_path)

        logging.info(f"Saved model state to {ms_path}\n")

    avg_epoch_time = ut.FormatSeconds(sum(epoch_time) / num_epochs)
    total_training_time = ut.FormatSeconds(
        ut.ElapsedSecondsSince(training_start_time))

    logging.info(f"Average epoch time: {avg_epoch_time}")
    logging.info(f"Total training time: {total_training_time}\n")

    tvl_path = os.path.join(output_dir, label + "_loss.csv")
    ut.SaveLoss2Csv(training_loss, validation_loss, tvl_path)
    logging.info(f"Saved training and validation loss to {tvl_path}\n")

    plt.figure(figsize=(10, 5))
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.title('Training/Validation Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    tvplot_path = os.path.join(output_dir, label + ".png")
    plt.savefig(tvplot_path)
    logging.info(f"Saved training and validation plot {tvplot_path}\n")

    plt.show()

    # * * * * * * * * * * * * * * * *
    # TEST
    # * * * * * * * * * * * * * * * *
    logging.info("Testing")
    test_loss = 0

    latent_variables_mu = []
    latent_variables_logvar = []

    model.eval()
    with torch.no_grad():
        for batch in test_data:
            reconstructed, mu, logvar = model(batch)
            latent_variables_mu.append(mu.cpu().numpy())
            latent_variables_logvar.append(logvar.cpu().numpy())
            loss = VAE.loss_function(reconstructed, batch, mu, logvar, beta)
            test_loss = loss.item()

    avg_test_loss = test_loss / len(test_data.dataset)
    logging.info(f"Test Loss: {avg_test_loss}\n")

    m_path = os.path.join(output_dir, label + "_overall_metrics.txt")
    ut.SaveOtherMetrics(
        metrics[0],
        metrics[1],
        avg_epoch_time,
        total_training_time,
        avg_test_loss,
        m_path)
    logging.info(f"Saved overall metrics to {m_path}\n")

    # * * * * * * * * * * * * * * * *
    # LATENT VARIABLES
    # * * * * * * * * * * * * * * * *
    lvmu_path = os.path.join(output_dir, label + "_latent_variables_mu.csv")
    ut.SaveLatentVariables2Csv(latent_variables_mu, lvmu_path)
    logging.info(f"Saved latent variables mean to {lvmu_path}")

    lvlogvar_path = os.path.join(output_dir,
                                 label + "_latent_variables_logvar.csv")
    ut.SaveLatentVariables2Csv(latent_variables_mu, lvlogvar_path)
    logging.info(f"Saved latent variables logvar to {lvlogvar_path}")


if __name__ == "__main__":
    main()
