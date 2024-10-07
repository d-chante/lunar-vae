import datetime
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import yaml


class Utils(object):

    def __init__(self):
        pass

    @staticmethod
    def Json2Profile(json_file):
        '''
        TBD
        '''
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
            temp_values = [val[1] for val in data['data']]
            return np.array(temp_values)

        except Exception as e:
            logging.error(
                f"Error trying to open JSON file {json_file} : ({e})")
            return None

    def LoadData(self, data_dir, batch_size=200, tensor=True):
        '''
        TBD
        '''
        profile_list = os.listdir(data_dir)
        profiles = []

        for profile in profile_list:
            p = self.Json2Profile(os.path.join(data_dir, profile))
            if p is not None:
                profiles.append(p)

        if tensor:
            data = np.stack(profiles, axis=0)
            np.random.shuffle(data)

            data = np.expand_dims(data, axis=1)

            remainder_data, test_data = train_test_split(
                data, test_size=0.1, random_state=42)

            train_data, validation_data = train_test_split(
                remainder_data, test_size=0.1, random_state=42)

            mean = np.mean(train_data)
            std = np.std(train_data)

            train_data = self.Normalize(train_data, mean, std)
            train_tensor = torch.tensor(train_data, dtype=torch.float32)
            train_loader = DataLoader(train_tensor, batch_size, shuffle=True)

            validation_data = self.Normalize(validation_data, mean, std)
            validation_tensor = torch.tensor(
                validation_data, dtype=torch.float32)
            validation_loader = DataLoader(
                validation_tensor, batch_size, shuffle=False)

            test_data = self.Normalize(test_data, mean, std)
            test_tensor = torch.tensor(test_data, dtype=torch.float32)
            test_loader = DataLoader(test_tensor, batch_size, shuffle=False)

            ret = [train_loader, validation_loader, test_loader], [mean, std]

        else:
            ret = profiles

        return ret

    @staticmethod
    def Normalize(data, mean, std):
        '''
        TBD - Z-score normalization
        '''
        return (data - mean) / std

    @staticmethod
    def GetConfig(config_filepath):
        '''
        TBD
        '''
        with open(config_filepath, 'r') as file:
            config = yaml.safe_load(file)

        dir_config = config.get('directories', {})
        training_config = config.get('training_parameters', {})

        return dir_config, training_config

    @staticmethod
    def GenerateFilename():
        '''
        TBD
        '''
        timestamp = datetime.datetime.now()
        return timestamp.strftime("vae_%Y-%m-%d_%Hh%Mm%Ss")

    @staticmethod
    def VisualizeProfile(json_file, dims=(10, 6)):
        '''
        TBD
        '''
        with open(json_file, 'r') as file:
            data = json.load(file)

        lunar_time = [item[0] for item in data['data']]
        temperature = [item[1] for item in data['data']]

        plt.figure(figsize=dims)
        plt.plot(lunar_time, temperature, marker='o', linestyle='-', color='b')

        plt.title(f"{os.path.basename(json_file).split('.')[0]}")
        plt.xlabel('Lunar Time [Hours since 0h00]')
        plt.ylabel('Temperature [K]')

        plt.xlim(left=0, right=24)
        plt.xticks(range(0, 25, 4))

        plt.ylim(bottom=0, top=450)

        plt.grid(True)
        plt.show()

    @staticmethod
    def ElapsedSecondsSince(timestamp):
        '''
        TBD
        '''
        delta_t = datetime.datetime.now() - timestamp
        total_seconds = int(delta_t.total_seconds())

        return total_seconds

    @staticmethod
    def FormatSeconds(total_seconds):
        '''
        TBD
        '''
        total_seconds = int(total_seconds)

        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds"

    @staticmethod
    def CreateDirectory(dirpath):
        '''
        TBD
        '''
        os.makedirs(dirpath, exist_ok=True)

    @staticmethod
    def SaveLoss2Csv(
            training_reconstruction_loss,
            training_kl_divergence,
            training_elbo_loss,
            validation_reconstruction_loss,
            validation_kl_divergence,
            validation_elbo_loss,
            filepath):
        df = pd.DataFrame(
            {
                'training_reconstruction_loss': training_reconstruction_loss,
                'training_kl_divergence': training_kl_divergence,
                'training_elbo_loss': training_elbo_loss,
                'validation_reconstruction_loss': validation_reconstruction_loss,
                'validation_kl_divergence': validation_kl_divergence,
                'validation_elbo_loss': validation_elbo_loss})
        df.to_csv(filepath, index=True)

    @staticmethod
    def SaveLatentVariables2Csv(array, filepath):
        '''
        TBD
        '''
        reshaped_array = array[0].reshape(-1, 4)
        df = pd.DataFrame(
            reshaped_array,
            columns=[
                'latent_variable_1',
                'latent_variable_2',
                'latent_variable_3',
                'latent_variable_4'])
        df.to_csv(filepath, index=True)

    @staticmethod
    def SaveMetrics(
        label,
        config_file,
        latent_variables,
        learning_rate,
        gamma,
        beta,
        dropout,
        num_epochs,
        batch_size,
        gpu,
        data_split,
        data_mean,
        data_std,
        final_learning_rate,
        average_epoch_time,
        total_training_time,
        training_loss,
        validation_loss,
        test_loss,
        filepath
    ):
        df = pd.DataFrame({
            'label': [label],
            'config_file': [config_file],
            'latent_variables': [latent_variables],
            'learning_rate': [learning_rate],
            'gamma': [gamma],
            'beta': [beta],
            'dropout': [dropout],
            'num_epochs': [num_epochs],
            'batch_size': [batch_size],
            'gpu': [gpu],
            'data_split': [data_split],
            'data_mean': [data_mean],
            'data_std': [data_std],
            'final_learning_rate': [final_learning_rate],
            'average_epoch_time': [average_epoch_time],
            'total_training_time': [total_training_time],
            'training_loss': [training_loss],
            'validation_loss': [validation_loss],
            'test_loss': [test_loss],
            'training_loss_kelvins': [training_loss * data_std],
            'validation_loss_kelvins': [validation_loss * data_std],
            'test_loss_kelvins': [test_loss * data_std]
        })

        file_exists = os.path.isfile(filepath)
        df.to_csv(
            filepath,
            mode='a' if file_exists else 'w',
            index=False,
            header=not file_exists)
