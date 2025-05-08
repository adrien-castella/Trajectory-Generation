import os
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from tqdm import tqdm
import shutil
from pathlib import Path

# Function to load specific CSV files from the Kaggle dataset
def load_kaggle_dataset(file_path: str):
    try:
        # Load the dataset using kagglehub
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "jefmenegazzo/pvs-passive-vehicular-sensors-datasets",  # Dataset ID
            file_path  # Path to the specific file in the dataset
        )
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


# Function to process and combine the vehicle data
def process_vehicle_data(vehicle_ID, data_folder):
    PATH = os.path.join(data_folder, f"PVS {vehicle_ID}")

    # Load and combine the input datasets
    columns_to_load_left = ['timestamp', 'acc_x_below_suspension', 'acc_y_below_suspension',
                            'acc_z_below_suspension', 'gyro_x_below_suspension',
                            'gyro_y_below_suspension', 'gyro_z_below_suspension',
                            'temp_dashboard', 'timestamp_gps', 'speed',
                            'latitude', 'longitude']
    columns_to_load_right = ['timestamp', 'acc_x_below_suspension', 'acc_y_below_suspension',
                             'acc_z_below_suspension', 'gyro_x_below_suspension',
                             'gyro_y_below_suspension', 'gyro_z_below_suspension',
                             'temp_dashboard']

    df_left = pd.read_csv(os.path.join(PATH, 'dataset_gps_mpu_left.csv'), usecols=columns_to_load_left)
    df_right = pd.read_csv(os.path.join(PATH, 'dataset_gps_mpu_right.csv'), usecols=columns_to_load_right)

    # Merge the two datasets on the 'timestamp' column
    df_combined = pd.merge(df_left, df_right, on='timestamp', suffixes=('_left', '_right'))

    # Load the classification dataset
    columns_to_load_labels = ['dirt_road', 'cobblestone_road', 'asphalt_road', 'no_speed_bump', 
                               'good_road_left', 'regular_road_left', 'bad_road_left', 
                               'good_road_right', 'regular_road_right', 'bad_road_right']
    df_labels = pd.read_csv(os.path.join(PATH, 'dataset_labels.csv'), usecols=columns_to_load_labels)

    # Combine the input data with the classification data
    df_final = pd.concat([df_combined, df_labels], axis=1)

    # Create filtered dataframe with specific columns
    columns_to_save = ['latitude', 'longitude', 'speed', 'dirt_road', 'cobblestone_road', 'timestamp_gps',
                       'asphalt_road', 'no_speed_bump', 'good_road_left', 'regular_road_left', 'bad_road_left', 
                       'good_road_right', 'regular_road_right', 'bad_road_right', 'timestamp']
    df_filtered = df_final[columns_to_save]

    # Keep only one data point for each unique 'timestamp_gps' where 'timestamp' is closest to it
    df_filtered = df_filtered.loc[df_filtered.groupby('timestamp_gps')['timestamp'].apply(lambda x: (x - x.name).abs().idxmin())]
    df_filtered = df_filtered.drop(columns=['timestamp'])

    # Create filtered dataframe with acc, gyro, and columns ending with _road
    columns_to_save_acc_gyro_road = [col for col in df_final.columns if 
                                     col.startswith('acc_') or col.startswith('gyro_') or col.endswith('_road') or col in ['speed']]
    columns_to_save_acc_gyro_road += ['latitude', 'longitude']
    df_filtered_acc_gyro_road = df_final[columns_to_save_acc_gyro_road]

    return df_final, df_filtered, df_filtered_acc_gyro_road


# Function to download the dataset and process it
def download_and_process_data(force_redownload=False):
    data_folder = Path("datasets/pvs")
    data_folder.mkdir(parents=True, exist_ok=True)

    # If force_redownload is True, remove existing data
    if force_redownload and os.path.exists(data_folder):
        shutil.rmtree(data_folder)  # Delete the existing data folder to force redownload
        os.makedirs(data_folder, exist_ok=True)

    # List of files we want to download from each PVS folder
    folders_to_download = [f'PVS {i}' for i in range(1, 10)]
    files_to_download = [
        "dataset_gps_mpu_left.csv",
        "dataset_gps_mpu_right.csv",
        "dataset_labels.csv",
    ]

    # Check if data is already downloaded
    data_downloaded = True
    for folder in folders_to_download:
        folder_path = os.path.join(data_folder, folder)
        if not os.path.exists(folder_path):
            data_downloaded = False
            break
    
    if not data_downloaded or force_redownload:
        # Iterate through the files and load them
        for folder in tqdm(folders_to_download, desc="Downloading and processing datasets", leave=False):
            os.makedirs(os.path.join(data_folder, folder), exist_ok=True)
            for file_name in files_to_download:
                file_path = os.path.join(folder, file_name)
                df = load_kaggle_dataset(file_path)
                if df is not None:
                    # Save each loaded dataframe if needed (optional)
                    df.to_csv(os.path.join(data_folder, file_path), index=False)

    # Check if preprocessing is needed
    preprocess_needed = True
    if os.path.exists('trajectory_prediction') and os.path.exists('road_prediction') and not force_redownload:
        preprocess_needed = False

    if preprocess_needed or force_redownload:
        # After downloading, process the vehicle data
        os.makedirs('trajectory_prediction', exist_ok=True)
        os.makedirs('road_prediction', exist_ok=True)

        for vehicle_ID in tqdm(range(1, 10), desc="Processing vehicle data"):
            # Process and save the data for each vehicle
            df_final, df_filtered, df_filtered_acc_gyro_road = process_vehicle_data(vehicle_ID, data_folder)

            pd.DataFrame(df_filtered).to_json(os.path.join('trajectory_prediction', f'{vehicle_ID}.json'), orient='records', lines=True)
            pd.DataFrame(df_filtered_acc_gyro_road).to_json(os.path.join('road_prediction', f'{vehicle_ID}.json'), orient='records', lines=True)

    print("Data processing complete.")