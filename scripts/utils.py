from datetime import datetime
import json
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import shutil
from scipy.signal import find_peaks
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from tqdm import tqdm
import yaml

from lunar_vae import Wrap1d


def GetConfig(config_filepath):
    '''
    @brief Returns configuration parameters from yaml file
    '''
    try:
        with open(config_filepath, 'r') as file:
            config = yaml.safe_load(file)
    except (FileNotFoundError, yaml.YAMLError, Exception) as e:
        print(f"Error: {e}")
        return None

    return (
        config.get('meta_config', {}),
        config.get('data_config', {}),
        config.get('training_config', {})
    )


def CopyConfigFile(config_filepath, target_dir):
    '''
    @brief Copies the config file to the target directory
    '''
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, os.path.basename(config_filepath))
    shutil.copy(config_filepath, target_path)


def PrintAndLog(log_filepath, message):
    '''
    @brief Prints a message and appends to a log file
    '''
    print(message.strip())
    with open(log_filepath, 'a') as file:
        file.write(message)


def PrintConfig(meta_config, data_config, training_config):
    print("Configuration:")
    print(f"\tDescription: {meta_config['description']}")
    print(f"\tInput Data Path: {data_config['input_data_path']}")
    print(f"\tOutput Directory: {data_config['output_dir']}")
    print(f"\tDevice: {training_config['device']}")
    print(f"\tLatent Dims: {training_config['latent_dim']}")
    print(f"\tBatch Size: {training_config['batch_size']}")
    print(f"\tEpochs: {training_config['epochs']}")
    print(f"\tBeta: {training_config['beta']}")
    print(f"\tLearning Rate: {training_config['learning_rate']}")
    print(f"\tGamma: {training_config['gamma']}")


def LoadTemperatureDataV1(input_csv_path, device):
    '''
    @brief Loads temperature data from profiles-v1
    @param input_csv_path The profile data in csv format
    @param device Target CPU or GPU device
    '''
    temp_data = []

    with open(input_csv_path, "r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Loading Profiles"):
            try:
                data = json.loads(line.strip())
                temps = data["data"]["temperature"]
                temp_data.append(temps)
            except (json.JSONDecodeError, KeyError) as e:
                tqdm.write(f"Skipping line due to error: {e}")

    temp_data = np.array(temp_data, dtype=np.float32)
    temp_data = torch.tensor(temp_data, dtype=torch.float32).to(device)

    return temp_data

def LoadTemperatureDataInRageV1(input_csv_path, device, range=[50, 425]):
    '''
    @brief Loads temperature data from profiles-v1 and returns 
        both temps and profile_list
    @param input_csv_path The profile data in csv format
    @param device Target CPU or GPU device
    '''
    temp_data = []
    profile_list = []

    with open(input_csv_path, "r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Loading Profiles"):
            try:
                data = json.loads(line.strip())
                temps = data["data"]["temperature"]
                profile = data["name"] + ".json"
                if all(range[0] <= temp <= range[1] for temp in temps):
                    temp_data.append(temps)
                    profile_list.append(profile)
            except (json.JSONDecodeError, KeyError) as e:
                tqdm.write(f"Skipping line due to error: {e}")

    temp_data = np.array(temp_data, dtype=np.float32)
    temp_data = torch.tensor(temp_data, dtype=torch.float32).to(device)

    return temp_data, profile_list


def LoadTemperatureDataV2(input_csv_path, device):
    '''
    @brief Loads temperature data from profiles-v2
    @param input_csv_path The profile data in csv format
    @param device Target CPU or GPU device
    '''
    temp_data = []

    with open(input_csv_path, "r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Loading Profiles"):
            try:
                data = json.loads(line.strip())
                temps = data["interpolated_data"]["temps"]
                temp_data.append(temps)
            except (json.JSONDecodeError, KeyError) as e:
                tqdm.write(f"Skipping line due to error: {e}")

    temp_data = np.array(temp_data, dtype=np.float32)
    temp_data = torch.tensor(temp_data, dtype=torch.float32).to(device)

    return temp_data


def LoadFilteredTemperatureDataV2(data_csv_path, device, mode="low"):
    '''
    @brief Loads profiles-v2 temperature data where temperatures in time
        range are below a temp boundary
    '''
    filtered_profiles = []
    filtered_profile_names = []

    with open(data_csv_path, "r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Filtering Data"):
            try:
                data = json.loads(line.strip())

                time_values = np.array(
                    data["interpolated_data"]["time"],
                    dtype=np.float32)
                temp_values = np.array(
                    data["interpolated_data"]["temps"],
                    dtype=np.float32)

                filter_conditions = []

                if mode == "low":
                    mask1 = (time_values >= 11) & (time_values <= 13)
                    filter_conditions.append(np.all(temp_values[mask1] > 300))
                    filter_conditions.append(np.all(temp_values[mask1] < 325))

                    mask2 = (time_values >= 9) & (time_values <= 15)
                    filter_conditions.append(np.all(temp_values[mask2] < 325))

                    mask3 = (time_values >= 0) & (time_values <= 1)
                    filter_conditions.append(np.all(temp_values[mask3] > 75))
                    filter_conditions.append(np.all(temp_values[mask3] < 100))

                    mask4 = (time_values >= 5) & (time_values <= 6)
                    filter_conditions.append(np.all(temp_values[mask4] > 100))
                elif mode == "high":
                    mask1 = (time_values >= 11) & (time_values <= 13)
                    filter_conditions.append(np.all(temp_values[mask1] > 370))
                    filter_conditions.append(np.all(temp_values[mask1] < 390))

                    mask2 = (time_values >= 9) & (time_values <= 15)
                    filter_conditions.append(np.all(temp_values[mask2] < 390))

                    mask3 = (time_values >= 0) & (time_values <= 1)
                    filter_conditions.append(np.all(temp_values[mask3] > 87.5))
                    filter_conditions.append(np.all(temp_values[mask3] < 115))

                    mask4 = (time_values >= 5) & (time_values <= 6)
                    filter_conditions.append(np.all(temp_values[mask4] < 115))
                else:
                    return

                if all(filter_conditions):
                    filtered_profiles.append(temp_values)
                    if data["name"] not in filtered_profile_names:
                        filtered_profile_names.append(data["name"])

            except (json.JSONDecodeError, KeyError) as e:
                tqdm.write(f"Skipping line due to error: {e}")
    if not filtered_profiles:
        raise ValueError("No valid filtered data found.")

    data_array = np.vstack(filtered_profiles)
    return (
        torch.tensor(data_array, dtype=torch.float32).to(device),
        filtered_profile_names
    )


def RandomSelectFromDict(target_dict, num_to_keep):
    '''
    @brief Randomly selects dictionary entries
    '''
    if len(target_dict) > num_to_keep:
        dict_to_list = list(target_dict.items())
        random_select = random.sample(dict_to_list, num_to_keep)
        return dict(random_select)
    else:
        return target_dict


def ConvertDictToLists(target_dict):
    keys = list(target_dict.keys())
    vals = list(target_dict.values())
    return vals, keys


def ReadCsvV2(csv_filepath):
    csv_data = []
    with open(csv_filepath, "r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Reading data"):
            try:
                data = json.loads(line.strip())
                data = {
                    "name": data["name"],
                    "time": data["interpolated_data"]["time"],
                    "temps": data["interpolated_data"]["temps"]
                }
                csv_data.append(data)
            except (json.JSONDecodeError, KeyError) as e:
                tqdm.write(f"Skipping line due to error: {e}")
    return csv_data


def CreateSubsetByTempInterval(csv_data, temp_range, time_range, num_to_keep):
    '''
    @brief tbd
    '''
    filtered_profiles = {}
    for data in tqdm(csv_data, desc="Filtering"):
        time_values = np.array(data["time"], dtype=np.float32)
        temp_values = np.array(data["temps"], dtype=np.float32)

        '''
		filter_conditions = []

		mask1 = (time_values >= time_range[0]) & (time_values <= time_range[1])
		filter_conditions.append(np.any(temp_values[mask1] > temp_range[0]))
		filter_conditions.append(np.any(temp_values[mask1] < temp_range[1]))

		if all(filter_conditions):
			filtered_profiles[data["name"]] = temp_values
		'''
        if temp_range[0] <= np.max(temp_values) <= temp_range[1]:
            filtered_profiles[data["name"]] = temp_values

    if not filtered_profiles:
        raise ValueError("No valid filtered data found.")

    filtered_profiles = RandomSelectFromDict(filtered_profiles, num_to_keep)
    temps, names = ConvertDictToLists(filtered_profiles)

    if len(temps) != len(names):
        raise ValueError("Profile temps and names are not equal")

    return (
        temps,
        names
    )


def GetDataMetrics(data):
    '''
    @brief Provides data metrics (mean/std/min/max)
    '''
    return (
        torch.mean(data),
        torch.std(data),
        torch.min(data),
        torch.max(data)
    )


def SplitAndNormalizeData(
        data,
        scaler,
        batch_size,
        device,
        split=[
            0.7,
            0.2,
            0.1]):
    '''
    @brief Splits data into train/val/test sets, and normalizes to the mean/std
        	of the training data. Expects shape (1, 120).
    '''
    dataset = TensorDataset(data, data)

    # Split data
    train_size = int(split[0] * len(dataset))
    val_size = int(split[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    torch.manual_seed(123)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size])

    # Normalize data
    train_data = torch.cat([data[0].unsqueeze(0)
                            for data in train_dataset], dim=0).numpy()
    val_data = torch.cat([data[0].unsqueeze(0)
                          for data in val_dataset], dim=0).numpy()
    test_data = torch.cat([data[0].unsqueeze(0)
                           for data in test_dataset], dim=0).numpy()

    train_data_norm = scaler.fit_transform(train_data)
    val_data_norm = scaler.transform(val_data)
    test_data_norm = scaler.transform(test_data)

    # Reshape to [batch_size, 1, 120]
    train_tensor = torch.tensor(train_data_norm,
                                dtype=torch.float32).unsqueeze(1).to(device)
    val_tensor = torch.tensor(val_data_norm,
                              dtype=torch.float32).unsqueeze(1).to(device)
    test_tensor = torch.tensor(test_data_norm,
                               dtype=torch.float32).unsqueeze(1).to(device)

    # Create DataLoaders
    train_dataset = TensorDataset(train_tensor, train_tensor)
    val_dataset = TensorDataset(val_tensor, val_tensor)
    test_dataset = TensorDataset(test_tensor, test_tensor)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )


@staticmethod
def SetupOutputDir(output_dir, parent_folder=None, sub_folders=[]):
    '''
    @brief Creates output directory
    '''
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if parent_folder is None:
        parent_folder = f'lunar_vae_{timestamp}'
    else:
        parent_folder = f'{parent_folder}_{timestamp}'
    model_dir = os.path.join(output_dir, parent_folder)
    os.makedirs(model_dir, exist_ok=True)
    for folder in sub_folders:
        os.makedirs(os.path.join(model_dir, folder), exist_ok=True)
    return model_dir


def PlotLosses(metrics, epochs, model_dir):
    '''
    @brief Generates plots to compare validation/training losses
    '''
    fig, axs = plt.subplots(3, 2, figsize=(12, 18))

    # Enforce  that the train/validation plots use the same y-axis scale
    kl_min, kl_max = min(min(metrics['train_kl_losses']), min(metrics['val_kl_losses'])), max(
        max(metrics['train_kl_losses']), max(metrics['val_kl_losses']))
    l1_min, l1_max = min(min(metrics['train_l1_losses']), min(metrics['val_l1_losses'])), max(
        max(metrics['train_l1_losses']), max(metrics['val_l1_losses']))
    l2_min, l2_max = min(min(metrics['train_l2_losses']), min(metrics['val_l2_losses'])), max(
        max(metrics['train_l2_losses']), max(metrics['val_l2_losses']))

    # KL Divergence Loss
    axs[0, 0].plot(range(1, epochs + 1), metrics['train_kl_losses'],
                   label='Train KL Divergence Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Training KL Divergence Loss')
    axs[0, 0].grid(True)
    axs[0, 0].set_ylim(kl_min, kl_max)

    axs[0, 1].plot(range(1, epochs + 1), metrics['val_kl_losses'],
                   label='Validation KL Divergence Loss')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_title('Validation KL Divergence Loss')
    axs[0, 1].grid(True)
    axs[0, 1].set_ylim(kl_min, kl_max)

    # L1 Loss
    axs[1, 0].plot(range(1, epochs + 1),
                   metrics['train_l1_losses'], label='Train L1 Loss')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].set_title('Training L1 Loss')
    axs[1, 0].grid(True)
    axs[1, 0].set_ylim(l1_min, l1_max)

    axs[1, 1].plot(range(1, epochs + 1), metrics['val_l1_losses'],
                   label='Validation L1 Loss')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].set_title('Validation L1 Loss')
    axs[1, 1].grid(True)
    axs[1, 1].set_ylim(l1_min, l1_max)

    # L2 Loss
    axs[2, 0].plot(range(1, epochs + 1),
                   metrics['train_l2_losses'], label='Train L2 Loss')
    axs[2, 0].set_xlabel('Epochs')
    axs[2, 0].set_ylabel('Loss')
    axs[2, 0].set_title('Training L2 Loss')
    axs[2, 0].grid(True)
    axs[2, 0].set_ylim(l2_min, l2_max)

    axs[2, 1].plot(range(1, epochs + 1), metrics['val_l2_losses'],
                   label='Validation L2 Loss')
    axs[2, 1].set_xlabel('Epochs')
    axs[2, 1].set_ylabel('Loss')
    axs[2, 1].set_title('Validation L2 Loss')
    axs[2, 1].grid(True)
    axs[2, 1].set_ylim(l2_min, l2_max)

    plt.tight_layout()
    plt.savefig(f"{model_dir}/vae_losses.png")
    plt.show()
    plt.close()


def SaveLossesToCSV(losses, csv_filepath):
    '''
    @brief Saves losses to a CSV file
    '''
    losses_df = pd.DataFrame(losses)
    losses_df = losses_df.reset_index()
    losses_df = losses_df.rename(columns={"index": "epochs"})
    losses_df.to_csv(csv_filepath, index=False)


def GetMeanLatentValues(z_sample, latent_dim):
    '''
    @brief Returns the Mean Latent Values for each
        dimension
    '''
    mean_vals = []
    for dim in range(latent_dim):
        mean_vals.append(np.mean(z_sample[:, dim]))
    return mean_vals


def PlotRawAndInterpolatedData(json_filepath):
    with open(json_filepath, 'r') as file:
        data = json.load(file)

    try:
        raw_data = data["raw_data"]
        interpolated_data = data["interpolated_data"]

        raw_time = raw_data["time"]
        raw_temps = raw_data["temps"]

        interp_time = interpolated_data["time"]
        interp_temps = interpolated_data["temps"]

    except KeyError as e:
        print(f"Missing key in JSON data: {e}")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(raw_time, raw_temps, 'o-', label='Raw Data', markersize=6)
    plt.plot(
        interp_time,
        interp_temps,
        '-',
        label='Interpolated Data',
        linewidth=1.5)
    plt.xlim(0, 24)
    plt.xticks([0, 6, 12, 18, 24])
    plt.ylim(25, 425)

    plt.title('Comparison of Raw and Interpolated Data')
    plt.xlabel('Time (Local Lunar Time)')
    plt.ylabel('Temperature (K)')
    plt.legend()

    plt.grid(True)
    plt.show()


def CreateListJsonFromDir(directory_path, verify=True):
    '''
    @brief Creates a list of all profile jsons in a directory
    '''
    verified_json_files = []
    json_files = sorted([f for f in os.listdir(
        directory_path) if f.endswith('.json')])
    for profile in tqdm(json_files, desc="Verifying Profiles"):
        if verify and verify_profile(os.path.join(directory_path, profile)):
            verified_json_files.append(profile)
        elif not verify:
            verified_json_files.append(profile)
    return verified_json_files


def CreateProfileListJson(output_path, json_list):
    '''
    @brief Saves a list of Profiles to a json file
    '''
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary = {
        "file_count": len(json_list),
        "files": json_list
    }
    with open(output_path, 'w') as outfile:
        json.dump(summary, outfile, indent=4)


def verify_profile(profile_path, min_acceptable_temp=40):
    '''
    @brief Checks that a profile is valid based on min temp
    '''
    try:
        with open(profile_path, 'r') as f:
            data = json.load(f)
        if ('interpolated_data' not in data or
                'temps' not in data['interpolated_data'] or
                not isinstance(data['interpolated_data']['temps'], list)):
            return False
        if any(
                temp < min_acceptable_temp for temp in data['interpolated_data']['temps']):
            return False
        return True
    except (json.JSONDecodeError, FileNotFoundError, TypeError):
        return False


def ConvertProfileListToCsv(profile_dir, profile_list, output_dir):
    '''
    @brief Reads a profile list, opens each profile json, and saves
        the json string to a line in a csv file.
    '''
    filename = f"{datetime.now().strftime('%Y%m%d')}-dataset.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as file:
        for profile in tqdm(profile_list, desc="Processing Profiles"):
            try:
                if ".json" in profile:
                    profile_name = profile.split(".")[0]
                with open(os.path.join(profile_dir, f"{profile_name}.json"), 'r', encoding="utf-8") as json_file:
                    data = json.load(json_file)
                    data["name"] = profile.split(".")[0]
                    json_str = json.dumps(data)
                    file.write(json_str + "\n")
            except Exception as e:
                print(f"Error processing {profile}: {e}")
    print(f"Saved {len(profile_list)} profiles to {filepath}")
    return filepath


def FilterBumpyProfiles(profile_data):
    '''
    @brief Filters out profiles that are detected as being
        bumpy (having multiple peaks)
    '''
    filtered = []
    for i, test_data in tqdm(enumerate(profile_data),
                             desc="Filtering out bumpy profiles"):
        temp_data = np.array(test_data["temps"])
        time_data = np.array(test_data["time"])

        mask = (time_data >= 6) & (time_data <= 18)
        filtered_temp_data = temp_data[mask]

        peaks, _ = find_peaks(filtered_temp_data)

        if len(peaks) > 1:
            pass
        else:
            filtered.append(test_data)
    return filtered


def GenerateDensityPlot(temp_data, output_dir=None):
    '''
    @brief Generates density plot from temperature data
    @param temp_data Profile temperature data
    @param output_dir Directory to save plot if not None
    '''
    temp_data = temp_data.unsqueeze(1)
    wrapper = Wrap1d(wrap_size=1)
    temp_data = wrapper(temp_data)

    num_profiles = temp_data.shape[0]

    y_fit = np.linspace(0, 24, 122)
    x_data = temp_data.squeeze().cpu().numpy()

    bin_x = np.linspace(0, 24, 122)
    bin_y = np.linspace(25, 475, 450)
    density, _, _ = np.histogram2d(
        np.tile(
            y_fit, x_data.shape[0]), x_data.flatten(), bins=[
            bin_x, bin_y])

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])

    ax1 = plt.subplot(gs[0])
    cax = plt.subplot(gs[1])

    log_density = np.log1p(density.T)
    vmin, vmax = 0, 12

    im = ax1.imshow(
        log_density,
        aspect="auto",
        origin="lower",
        extent=[0, 24, 25, 475],
        cmap="plasma",
        vmin=vmin,
        vmax=vmax
    )

    ax1.set_title(f"Density Plot of Profiles (num_profiles = {num_profiles})")
    ax1.set_xlabel("Local lunar time (hours)")
    ax1.set_ylabel("Temperature (K)")
    ax1.set_xlim(0, 24)
    ax1.set_xticks([0, 6, 12, 18, 24])
    ax1.set_xticks(
        np.setdiff1d(
            np.arange(
                0, 25, 1), np.arange(
                0, 25, 6)), minor=True)
    ax1.set_ylim(25, 475)
    ax1.set_yticks(np.arange(50, 500, 50))
    ax1.set_yticks(
        np.setdiff1d(
            np.arange(
                25, 500, 25), np.arange(
                50, 500, 50)), minor=True)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.3)
    fig.colorbar(im, cax=cax, label="Density (log scale)")

    plt.tight_layout()

    if output_dir is not None:
        path = os.path.join(output_dir, "density_plot.png")
        plt.savefig(path)
        print(f"Saved figure to {path}")

    plt.show()
    plt.close()
    

def LoadProfileList(json_filepath):
    '''
    @brief Loads a list of Profile file names from a Profile
        list json
    @param json_filepath Path to the Profile list json
    '''
    try:
        with open(json_filepath, 'r') as file:
            data = json.load(file)
      
        if "files" in data:
            filenames = data["files"]
            if isinstance(filenames, list):
                return filenames
            else:
                raise ValueError("'files' field is not a list.")
        else:
            raise KeyError("'files' field is missing in the JSON file.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
def CollectStatisticsFromProfilesV1(profile_dir, profile_list, exclude_keywords=None):
    '''
    @brief Reads and aggregates statistics from list of Profiles
    @param profile_dir The parent directory of Profile files
    @param profile_list A json containing a list of target Profile files
    @param exclude_keyword Excludes files with keywords in the name
    '''
    aggregated_stats = {
        "max_temp": [],
        "min_temp": [],
        "mean_temp": [],
        "std_temp": []
    }

    exclude_keywords = exclude_keywords or []

    for filename in tqdm(profile_list, desc="Collecting profile statistics"):
        if any(keyword in filename for keyword in exclude_keywords):
            continue
        
        file_path = os.path.join(profile_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if "statistics" in data:
                stats = data["statistics"]
                aggregated_stats["max_temp"].append(stats.get("max_temp"))
                aggregated_stats["min_temp"].append(stats.get("min_temp"))
                aggregated_stats["mean_temp"].append(stats.get("mean_temp"))
                aggregated_stats["std_temp"].append(stats.get("std_temp"))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return aggregated_stats
    
def CollectStatisticsFromProfilesV2(profile_dir, profile_list, exclude_keywords=None):
    '''
    @brief Reads and aggregates statistics from list of Profiles
    @param profile_dir The parent directory of Profile files
    @param profile_list A json containing a list of target Profile files
    @param exclude_keyword Excludes files with keywords in the name
    '''
    aggregated_stats = {
        "max_temp": [],
        "min_temp": [],
        "mean_temp": [],
        "std_temp": []
    }

    exclude_keywords = exclude_keywords or []

    for filename in tqdm(profile_list, desc="Collecting profile statistics"):
        if any(keyword in filename for keyword in exclude_keywords):
            continue
        
        file_path = os.path.join(profile_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if "interpolated_data" in data:
                stats = data["interpolated_data"]["statistics"]
                aggregated_stats["max_temp"].append(stats.get("max_temp"))
                aggregated_stats["min_temp"].append(stats.get("min_temp"))
                aggregated_stats["mean_temp"].append(stats.get("mean_temp"))
                aggregated_stats["std_temp"].append(stats.get("std_temp"))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return aggregated_stats


def PlotHistograms(profile_statistics, num_bins=50, output_dir=None):
    '''
    @brief Generates a plot for Profile dataset mean, standard deviation,
        min temp, and max temp
    @param profile_statistics A dictionary containing a list of Profile statistics 
    @param num_bins The number of bins to use
    '''
    for stat, values in profile_statistics.items():
        plt.hist(values, bins=num_bins, alpha=0.7, label=stat)
        plt.title(f'Histogram of {stat.replace("_", " ").title()}')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Frequency')
        plt.grid(True)
        if output_dir is not None:
            path = os.path.join(output_dir, f"histogram_{stat}.png")
            plt.savefig(path)
            print(f"Saved figure to {path}")
        plt.show()
        plt.close()


def PlotComparativeHistograms(
        profile_statistics_a, profile_statistics_b, labels=("Dataset A", "Dataset B"), output_dir=None):
    '''
    @brief Generates plots to compare Profile data set statistics
    @param profile_statistics_a First dataset
    @param profile_statistics_b Second dataset, usually a subset
    @param labels Label for datasets
    '''
    for stat in profile_statistics_a.keys():
        values_1 = profile_statistics_a[stat]
        values_2 = profile_statistics_b[stat]
        
        plt.hist(values_1, bins=50, alpha=0.7, label=f'{labels[0]} ({stat})', color='red')
        plt.hist(values_2, bins=50, alpha=0.7, label=f'{labels[1]} ({stat})', color='blue')
        
        plt.title(f'Comparative Histogram of {stat.replace("_", " ").title()}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        if output_dir is not None:
            path = os.path.join(output_dir, f"histogram_{stat}.png")
            plt.savefig(path)
            print(f"Saved figure to {path}")
        plt.show()
        plt.close()


def ComputeMetricRangesV1(profile_dir, profile_list):
    '''
    @brief Computes the statistic range over a Profile dataset
    @param profile_dir The path to the Profile directory
    @param profile_list A list of Profile filenames
    '''
    ranges = {
        "mean_temp": {"min": float('inf'), "max": float('-inf')},
        "max_temp": {"min": float('inf'), "max": float('-inf')},
        "min_temp": {"min": float('inf'), "max": float('-inf')},
        "std_temp": {"min": float('inf'), "max": float('-inf')}
    }

    for filename in tqdm(profile_list, desc="Computing metric ranges"):
        file_path = os.path.join(profile_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if "statistics" in data:
                stats = data["statistics"]
                for metric in ranges:
                    value = stats.get(metric)
                    if value is not None:
                        ranges[metric]["min"] = min(ranges[metric]["min"], value)
                        ranges[metric]["max"] = max(ranges[metric]["max"], value)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return {metric: (r["min"], r["max"]) for metric, r in ranges.items()}

def ComputeMetricRangesV2(profile_dir, profile_list):
    '''
    @brief Computes the statistic range over a Profile dataset
    @param profile_dir The path to the Profile directory
    @param profile_list A list of Profile filenames
    '''
    ranges = {
        "mean_temp": {"min": float('inf'), "max": float('-inf')},
        "max_temp": {"min": float('inf'), "max": float('-inf')},
        "min_temp": {"min": float('inf'), "max": float('-inf')},
        "std_temp": {"min": float('inf'), "max": float('-inf')}
    }

    for filename in tqdm(profile_list, desc="Computing metric ranges"):
        file_path = os.path.join(profile_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if "interpolated_data" in data:
                stats = data["interpolated_data"]["statistics"]
                for metric in ranges:
                    value = stats.get(metric)
                    if value is not None:
                        ranges[metric]["min"] = min(ranges[metric]["min"], value)
                        ranges[metric]["max"] = max(ranges[metric]["max"], value)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return {metric: (r["min"], r["max"]) for metric, r in ranges.items()}

def CreateBinnedProfileJsonV1(profile_dir, profile_list, output_filepath, ranges, num_bins=50):
    '''
        @brief Creates a json file that bins Profiles
        @param profile_dir The path to the Profile directory
        @param profile_list A list of Profile filenames
        @param output_filepath The path to the output json file
        @param ranges The upper and lower metric ranges 
        @param num_bins The number of bins
    '''
    binned_files = {
        "mean_temp": [[] for _ in range(num_bins)],
        "max_temp": [[] for _ in range(num_bins)],
        "min_temp": [[] for _ in range(num_bins)],
        "std_temp": [[] for _ in range(num_bins)],
    }

    bin_width = {
        metric: (ranges[metric][1] - ranges[metric][0]) / num_bins for metric in ranges
    }

    for filename in tqdm(profile_list, desc="Binning files"):
        file_path = os.path.join(profile_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if "statistics" in data:
                stats = data["statistics"]
                for metric in binned_files:
                    value = stats.get(metric)
                    if value is not None:
                        bin_index = min(
                            int(np.floor((value - ranges[metric][0]) / bin_width[metric])), 
                            num_bins - 1
                        )
                        binned_files[metric][bin_index].append(filename)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    with open(output_filepath, 'w') as outfile:
        json.dump(binned_files, outfile, indent=4)

def CreateBinnedProfileJsonV2(profile_dir, profile_list, output_filepath, ranges, num_bins=50):
    '''
        @brief Creates a json file that bins Profiles
        @param profile_dir The path to the Profile directory
        @param profile_list A list of Profile filenames
        @param output_filepath The path to the output json file
        @param ranges The upper and lower metric ranges 
        @param num_bins The number of bins
    '''
    binned_files = {
        "mean_temp": [[] for _ in range(num_bins)],
        "max_temp": [[] for _ in range(num_bins)],
        "min_temp": [[] for _ in range(num_bins)],
        "std_temp": [[] for _ in range(num_bins)],
    }

    bin_width = {
        metric: (ranges[metric][1] - ranges[metric][0]) / num_bins for metric in ranges
    }

    for filename in tqdm(profile_list, desc="Binning files"):
        file_path = os.path.join(profile_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if "interpolated_data" in data:
                stats = data["interpolated_data"]["statistics"]
                for metric in binned_files:
                    value = stats.get(metric)
                    if value is not None:
                        bin_index = min(
                            int(np.floor((value - ranges[metric][0]) / bin_width[metric])), 
                            num_bins - 1
                        )
                        binned_files[metric][bin_index].append(filename)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    with open(output_filepath, 'w') as outfile:
        json.dump(binned_files, outfile, indent=4)

def CreateProfileSubset(binned_list_path, output_filepath, metric="std_temp", sample_size=100000):
    '''
    @brief Generates a json containing a subset of Profiles based on metric and number of
        samples from each bin
    @param binned_list_path The binned json filepath
    @param output_filepath The target output json file
    @param metric The metric to bin on
    @param sample_size Max number of Profiles to sample from each bin
    '''
    with open(binned_list_path, 'r') as f:
        binned_data = json.load(f)
    
    if metric not in binned_data:
        raise ValueError(f"Metric '{metric}' not found in the binned list.")
    
    selected_files = []
    for file_list in binned_data[metric]:
        if len(file_list) > sample_size:
            selected_files.extend(random.sample(file_list, sample_size))
        else:
            selected_files.extend(file_list)
    
    output_data = {
        "count": len(selected_files),
        "files": selected_files
    }
    
    with open(output_filepath, 'w') as f:
        json.dump(output_data, f, indent=4)