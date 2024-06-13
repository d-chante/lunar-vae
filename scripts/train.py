import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from torchsummary import summary

from lunar_vae import VAE
from utils import Utils

ut = Utils()

# * * * * * * * * * * * * * * * * 
# PARAMETERS
# * * * * * * * * * * * * * * * * 
cfg_filepath = "/lunar-vae/config/cosmocanyon_cfg.yaml"
dirs, config = ut.GetConfig(cfg_filepath)

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
# DEVICE
# * * * * * * * * * * * * * * * * 
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
logging.info(f"Training with: {device}")

# * * * * * * * * * * * * * * * * 
# MODEL
# * * * * * * * * * * * * * * * * 
model = VAE(latent_variables).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
summary(model, input_dims, batch_size)
time.sleep(1)  # Wait a moment to print summary

# * * * * * * * * * * * * * * * * 
# DATA
# * * * * * * * * * * * * * * * * 
train_data, validation_data, test_data = ut.LoadData(profiles_dir, batch_size)

logging.info(f"Training data: {len(train_data.dataset)}")
logging.info(f"Validation data: {len(validation_data.dataset)}")
logging.info(f"Test data: {len(test_data.dataset)}")

# * * * * * * * * * * * * * * * * 
# TRAIN/VALIDATE
# * * * * * * * * * * * * * * * * 
training_loss = []
validation_loss = []

for epoch in range(num_epochs):
    logging.info(f"Epoch {epoch} of {num_epochs}")

    model.train()
    total_training_loss = 0
    for batch in train_data:
        batch = batch.to(device)
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
            batch = batch.to(device)
            reconstructed, mu, logvar = model(batch)
            loss = VAE.loss_function(reconstructed, batch, mu, logvar, beta)
            total_validation_loss += loss.item()

    avg_validation_loss = total_validation_loss / len(validation_data.dataset)
    validation_loss.append(avg_validation_loss)
    logging.info(f"Validation Loss: {avg_validation_loss}\n")

plt.figure(figsize=(10, 5))
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.title('Training/Validation Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
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
        batch = batch.to(device)
        reconstructed, mu, logvar = model(batch)
        latent_variables_mu.append(mu.cpu().numpy())
        latent_variables_logvar.append(logvar.cpu().numpy())
        loss = VAE.loss_function(reconstructed, batch, mu, logvar, beta)
        test_loss = loss.item()

avg_test_loss = test_loss / len(test_data.dataset)
print(f"Average Test Loss: {avg_test_loss}\n")

# * * * * * * * * * * * * * * * * 
# LATENT VARIABLES
# * * * * * * * * * * * * * * * * 

# * * * * * * * * * * * * * * * * 
# SAVE
# * * * * * * * * * * * * * * * * 

# TODO