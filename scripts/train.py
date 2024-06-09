import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import KFold
import time
import torch
from torch.utils.data import DataLoader, Subset
from torchsummary import summary

from lunar_vae import VAE
from utils import Utils

ut = Utils()

# Set up configuration parameters
cfg_filepath = "/lunar-vae/config/cosmocanyon_cfg.yaml"
dirs, config = ut.GetConfig(cfg_filepath)

profiles_dir = dirs['profiles_directory']
output_dir = dirs['output_directory']

latent_variables = config['latent_variables']
learning_rate = float(config['learning_rate'])
epochs = config['epochs']
batch_size = config['batch_size']
n_splits = config['n_splits']
gpu = config['gpu']
input_dims = (1, 120)

# Configure logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# Set up model and devices
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
logging.info(f"Training with: {device}")

vae = VAE(latent_variables).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
summary(vae, input_dims)
time.sleep(1)  # Wait a moment to print summary

data, data_test = ut.LoadData(profiles_dir)

kfold = KFold(n_splits, shuffle=True)

all_train_losses = []
all_val_losses = []

start_t = datetime.datetime.now()

for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
    logging.info(f"Fold {fold+1}/{n_splits}")
    fold_start_t = datetime.datetime.now()

    train_losses = []
    val_losses = []

    train_subset = Subset(data, train_idx)
    val_subset = Subset(data, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    vae.train()

    for epoch in range(epochs):
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            recon_batch, mu, logvar = vae(batch)
            loss = VAE.loss_function(recon_batch, batch, mu, logvar, beta=0.2)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}")

    vae.eval()

    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon_batch, mu, logvar = vae(batch)
            loss = VAE.loss_function(recon_batch, batch, mu, logvar, beta=0.2)
            val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        logging.info(f"Validation Loss for Fold {fold+1}: {avg_val_loss}")

    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)

    logging.info(f"Fold training time: {ut.ElapsedTimeSince(fold_start_t)}\n")

# Output results
logging.info(f"Training time: {ut.ElapsedTimeSince(start_t)}\n")

avg_train_losses = np.mean(all_train_losses, axis=0)
avg_val_losses = np.mean(all_val_losses, axis=0)

plt.figure(figsize=(8, 4))
plt.plot(avg_train_losses, label='Average Training Loss')
plt.plot(avg_val_losses, 'ro-', label='Average Validation Loss')
plt.title('Training vs Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save results to file
output_filename = os.path.join(output_dir, ut.GenerateFilename())

# Flatten the nested lists
flat_train_losses = [loss for sublist in all_train_losses for loss in sublist]
flat_val_losses = [loss for sublist in all_val_losses for loss in sublist]

# Ensure the lists are of equal length
assert len(flat_train_losses) == len(flat_val_losses), "Mismatch in lengths of training and validation losses"

# Write to CSV
with open(output_filename + ".csv", 'w') as file:
    file.write('Training Loss,Validation Loss\n')
    for train_loss, val_loss in zip(flat_train_losses, flat_val_losses):
        file.write(f"{train_loss},{val_loss}\n")
logging.info(f"Results saved to: {output_filename}.csv")

plt.savefig(output_filename + ".png", format='png', dpi=300)
logging.info(f"Plot saved to: {output_filename}.png")

model_state = vae.state_dict()
torch.save(model_state, output_filename + ".pth")
logging.info(f"Model saved to: {output_filename}.pth")
