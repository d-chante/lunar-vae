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

    def LoadData(self, data_dir, batch_size):
        '''
        TBD
        '''
        profile_list = os.listdir(data_dir)
        profiles = []

        for profile in profile_list:
            p = self.Json2Profile(os.path.join(data_dir, profile))
            if p is not None:
                profiles.append(p)

        data = np.stack(profiles, axis=0)

        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean) / std

        data_min = np.min(data)
        data_max = np.max(data)
        data = (data - data_min) / (data_max - data_min)

        data = np.expand_dims(data, axis=1)

        remainder_data, test_data = train_test_split(
            data, test_size=0.1, random_state=42)

        train_data, validation_data = train_test_split(
            remainder_data, test_size=0.2, random_state=42)

        train_tensor = torch.tensor(train_data, dtype=torch.float32)
        train_loader = DataLoader(train_tensor, batch_size, shuffle=True)

        validation_tensor = torch.tensor(validation_data, dtype=torch.float32)
        validation_loader = DataLoader(
            validation_tensor, batch_size, shuffle=True)

        test_tensor = torch.tensor(test_data, dtype=torch.float32)
        test_loader = DataLoader(test_tensor, batch_size, shuffle=True)

        return [train_loader, validation_loader, test_loader], [mean, std]

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
    def SaveLoss2Csv(training_loss, validation_loss, filepath):
        df = pd.DataFrame({'training_loss': training_loss,
                          'validation_loss': validation_loss})
        df.to_csv(filepath, index=False)

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
        df.to_csv(filepath, index=False)

    @staticmethod
    def SaveOtherMetrics(
            data_split,
            data_mean,
            data_std,
            final_learning_rate,
            average_epoch_time,
            total_training_time,
            test_loss,
            filepath):
        with open(filepath, 'w') as file:
            file.write(
                f"Data split [train/val/test]: {data_split[0]}/{data_split[1]}/{data_split[2]}\n")
            file.write(f"Data mean: {data_mean}\n")
            file.write(f"Data standard deviation: {data_std}\n")
            file.write(f"Final learning rate: {final_learning_rate}\n")
            file.write(f"Average epoch time: {average_epoch_time}\n")
            file.write(f"Total training time: {total_training_time}\n")
            file.write(f"Test loss: {test_loss}\n")
