import os, torch, multiprocessing
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import Dataset
import os

class TrajectoryDataset(Dataset):
    def __init__(self, file_path, n_samples=1000, sequence_length=10, noise_std=0.03, batch_size=1000, regenerate=False):
        self.file_path = file_path

        if regenerate or not os.path.exists(file_path):
            print(f"Generating new dataset and saving to {file_path} 💾")
            data = self._generate_dataset(n_samples, sequence_length, noise_std, batch_size)
            np.save(file_path, data)
        else:
            print(f"Loading dataset from {file_path} 📂")

        self.data = np.load(file_path, mmap_mode='r')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)

    @staticmethod
    def _generate_dataset(n_samples, seq_len, noise_std, batch_size):
        all_data = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            size = end - start

            # Time vector
            t = np.linspace(0, 1, seq_len)[None, :]  # shape (1, T)

            # Choose motion types
            motion_ids = np.random.randint(0, 3, size=size)  # 0=linear, 1=sine, 2=circular
            motion_onehots = np.eye(3)[motion_ids]  # (N, 3)

            # Expand to sequences
            motion_onehots_seq = np.repeat(motion_onehots[:, None, :], seq_len, axis=1)  # (N, T, 3)

            # Generate x, y according to type
            t_expanded = np.tile(t, (size, 1))  # (N, T)

            x = np.zeros((size, seq_len))
            y = np.zeros((size, seq_len))

            linear_mask = motion_ids == 0
            sine_mask = motion_ids == 1
            circular_mask = motion_ids == 2

            x[linear_mask] = t_expanded[linear_mask]
            y[linear_mask] = t_expanded[linear_mask]

            x[sine_mask] = t_expanded[sine_mask]
            y[sine_mask] = np.sin(2 * np.pi * t_expanded[sine_mask])

            x[circular_mask] = np.cos(2 * np.pi * t_expanded[circular_mask])
            y[circular_mask] = np.sin(2 * np.pi * t_expanded[circular_mask])

            # Add Gaussian noise
            x += np.random.normal(0, noise_std, size=(size, seq_len))
            y += np.random.normal(0, noise_std, size=(size, seq_len))

            # Compute speed (approximate)
            dx = np.diff(x, axis=1, prepend=x[:, :1])
            dy = np.diff(y, axis=1, prepend=y[:, :1])
            speed = np.sqrt(dx**2 + dy**2)
            speed += np.random.normal(0, noise_std * 0.5, size=(size, seq_len))

            # Generate obstacle one-hot (2 categories)
            obstacle = np.random.randint(0, 2, size=(size, seq_len))
            obstacle_onehot = np.zeros((size, seq_len, 2))
            obstacle_onehot[np.arange(size)[:, None], np.arange(seq_len)[None, :], obstacle] = 1

            # Stack all features: x, y, speed, motion (3), obstacle (2)
            features = np.concatenate([
                x[:, :, None], y[:, :, None], speed[:, :, None],
                motion_onehots_seq, obstacle_onehot
            ], axis=-1)  # shape: (N, T, 8)

            all_data.append(features)

        return np.concatenate(all_data, axis=0)

class ArgoverseDataset(Dataset):
    def __init__(self, split='train', cache_root="ArgoverseCache", num_agents=50, regenerate_cache=False):
        assert split in ['train', 'val', 'test'], "Split must be one of: train, val, test"
        assert num_agents >= 2, "num_agents must be at least 2 to fit AV and AGENT"

        raw_data_root = {
            'train': "datasets\\argoverse_v1.1\\extracted\\forecasting_train\\train\\data",
            'val': "datasets\\argoverse_v1.1\\extracted\\forecasting_val\\val\\data",
            'test': "datasets\\argoverse_v1.1\\extracted\\forecasting_test\\test_obs\\data"
        }

        self.split = split
        self.num_agents = num_agents
        self.cache_root = cache_root
        self.raw_data_path = raw_data_root[split]
        self.cache_dir = os.path.join(cache_root, f"{split}_agents{num_agents}")
        self.cache_file = os.path.join(self.cache_dir, 'features.npy')

        os.makedirs(self.cache_dir, exist_ok=True)

        if not os.path.exists(self.cache_file) or regenerate_cache:
            print(f"🛠️ Generating {split} cache (num_agents={num_agents})...")
            self._preprocess_and_cache()
        else:
            print(f"📂 Using cached {split} dataset with {num_agents} agents.")

        self.features = np.load(self.cache_file, mmap_mode='r')
        self.num_samples = self.features.shape[0]

        statistics = np.load(os.path.join(cache_root, 'parameters.npz'))
        self.mean, self.std = statistics['mean'], statistics['std']

    @staticmethod
    def _make_data_static(path, num_agents):
        df = pd.read_csv(path)
        timestamps = df['TIMESTAMP'].unique()
        num_steps = len(timestamps)

        def make_matrix(id):
            output_matrix = np.zeros((50, 4))
            filtered_df = df[df['TRACK_ID'] == id].reset_index(drop=True)

            time_to_row = {filtered_df['TIMESTAMP'].iloc[i]:
                           (filtered_df['X'].iloc[i], filtered_df['Y'].iloc[i])
                           for i in range(len(filtered_df))}

            for idx, timestamp in enumerate(timestamps):
                if timestamp in time_to_row:
                    output_matrix[idx, 0] = time_to_row[timestamp][0]
                    output_matrix[idx, 1] = time_to_row[timestamp][1]
                    output_matrix[idx, 2] = timestamp
                else:
                    output_matrix[idx, 3] = 1  # mask if timestamp missing

            return output_matrix

        data = np.zeros((num_agents, 50, 4))
        data[:, :, -1] = 1

        av_id = df[df['OBJECT_TYPE'] == 'AV']['TRACK_ID'].unique()[0]
        data[0] = make_matrix(av_id)

        agent_id = df[df['OBJECT_TYPE'] == 'AGENT']['TRACK_ID'].unique()[0]
        data[1] = make_matrix(agent_id)

        other_ids = df[df['OBJECT_TYPE'] == 'OTHERS']['TRACK_ID'].unique()
        max_other = num_agents - 2

        for i in range(min(len(other_ids), max_other)):
            data[i + 2] = make_matrix(other_ids[i])
        
        max_i = (num_steps - 20) // 15 + 1
        data_list = [data[:, i*15 : 20 + i*15] for i in range(0, max_i)]

        return data_list

    def _preprocess_and_cache(self):
        file_names = os.listdir(self.raw_data_path)
        full_paths = [os.path.join(self.raw_data_path, fn) for fn in file_names]
        dataset = []

        num_workers = max(multiprocessing.cpu_count() - 1, 1)
        print(f"🚀 Using {num_workers} workers to process {len(file_names)} files in parallel...")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(ArgoverseDataset._make_data_static, path, self.num_agents): path
                for path in full_paths
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Parallel Processing {self.split}"):
                try:
                    dataset.extend(future.result())
                except Exception as e:
                    print(f"❌ Failed on file {futures[future]}: {e}")

        dataset = np.stack(dataset)

        if self.split == 'train':
            mean = dataset.mean(axis=0)
            std = dataset.std(axis=0)

            mean[:, :, -1] = 0
            std[:, :, -1] = 1

            std[std == 0] = 1
            
            np.savez_compressed(os.path.join(self.cache_root, 'parameters.npz'), mean=mean, std=std)
        else:
            statistics = np.load(os.path.join(self.cache_root, 'parameters.npz'))
            mean, std = statistics['mean'], statistics['std']

        dataset = (dataset - mean) / std
        
        np.save(self.cache_file, dataset.astype(np.float32))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'input': torch.from_numpy(self.features[idx]).float()
        }

class RoadPredictionDataset(Dataset):
    def __init__(self, df_paths, sequence_length=50, cache_dir='cached_road', regenerate_cache=False):
        self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)
        self.sequence_files = []

        if regenerate_cache or not self._is_cache_ready():
            print("Generating cached sequences...")
            self._prepare_and_cache(df_paths, sequence_length)
            self._load_parameters()
        else:
            print("Using cached sequences.")
            self._load_parameters()
    
    def _is_cache_ready(self):
        return all(i in os.listdir(self.cache_dir) for i in ['features.npy', 'targets.npy', 'conditions.npy', 'parameters.npz'])
    
    def _prepare_and_cache(self, df_paths, seq_len=50):
        all_features = []
        all_targets = []
        all_conditions = []

        for file_path in df_paths:
            df = pd.read_json(file_path, lines=True)
            if df.empty:
                continue

            features = df.drop(columns=['asphalt_road', 'dirt_road', 'cobblestone_road', 'latitude', 'longitude']).values
            targets = df[['asphalt_road', 'dirt_road', 'cobblestone_road']].values
            conditions = df[['latitude', 'longitude']].values

            for i in range(len(features) - seq_len):
                seq_features = features[i:i + seq_len].copy()
                seq_target = targets[i + seq_len - 1].copy()
                seq_condition = conditions[i].copy()

                seq_targets = targets[i:i + seq_len]
                if np.abs(seq_targets - seq_target).sum() == 0:
                    all_features.append(seq_features)
                    all_targets.append(seq_target)
                    all_conditions.append(seq_condition)

        all_features = np.stack(all_features)
        all_targets = np.stack(all_targets)
        all_conditions = np.stack(all_conditions)

        # Normalize
        self.mean = all_features.mean(axis=0)
        self.std = all_features.std(axis=0)

        self.cond_mean = all_conditions.mean(axis=0)
        self.cond_std = all_conditions.std(axis=0)

        all_features = (all_features - self.mean) / self.std
        all_conditions = (all_conditions - self.cond_mean) / self.cond_std

        # Save statistics
        np.savez(os.path.join(self.cache_dir, 'parameters.npz'),
                 mean=self.mean, std=self.std,
                 cond_mean=self.cond_mean, cond_std=self.cond_std)

        # Save as memmap .npy
        feature_path = os.path.join(self.cache_dir, 'features.npy')
        target_path = os.path.join(self.cache_dir, 'targets.npy')
        condition_path = os.path.join(self.cache_dir, 'conditions.npy')

        np.save(feature_path, all_features.astype(np.float32))
        np.save(target_path, all_targets.astype(np.float32))
        np.save(condition_path, all_conditions.astype(np.float32))

        self.num_samples = all_features.shape[0]
    
    def _load_parameters(self):
        stats = np.load(os.path.join(self.cache_dir, 'parameters.npz'))
        self.mean = stats['mean']
        self.std = stats['std']
        self.cond_mean = stats['cond_mean']
        self.cond_std = stats['cond_std']

        self.features = np.load(os.path.join(self.cache_dir, 'features.npy'), mmap_mode='r')
        self.targets = np.load(os.path.join(self.cache_dir, 'targets.npy'), mmap_mode='r')
        self.conditions = np.load(os.path.join(self.cache_dir, 'conditions.npy'), mmap_mode='r')

        self.num_samples = self.features.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'input': torch.from_numpy(self.features[idx]).float(),
            'label': torch.from_numpy(self.targets[idx]).float(),
            'cond': torch.from_numpy(self.conditions[idx]).float()
        }

class TrajectoryPredictionDataset(Dataset):
    def __init__(self, df_paths, sequence_length=10):
        self.sequence_length = sequence_length
        self.sequences = []

        for file_path in df_paths:
            df = pd.read_json(file_path, lines=True)
            if len(df) == 0:
                continue

            # Turn latitude and longitude into differences
            df[['latitude_diff', 'longitude_diff']] = df[['latitude', 'longitude']].diff()
            df = df.iloc[1:].reset_index(drop=True)

            # Create two columns for no_speed_bump
            no_speed_bump = df['no_speed_bump'].values
            second_column = abs(no_speed_bump - 1)
            
            # Get features and insert the second column
            features = df.drop(columns=['timestamp_gps']).values
            self.features = np.insert(features, 7, second_column, axis=1)
            
            self.timestamps = df['timestamp_gps'].values
            self.targets = df[['latitude', 'longitude']].values

            self.sequences.extend(self._create_sequences())
        
        all_inputs = np.stack([seq[0][:, :5] for seq in self.sequences])
        self.mean = np.mean(all_inputs, axis=0)
        self.std = np.std(all_inputs, axis=0)
        self.std[self.std == 0] = 1

        all_targets = np.stack([seq[1] for seq in self.sequences])
        self.t_mean = np.mean(all_targets, axis=0)
        self.t_std = np.std(all_targets, axis=0)
        self.t_std[self.t_std == 0] = 1
        
        self._normalize_sequences()
    
    def _normalize_sequences(self):
        for i in range(len(self.sequences)):
            seq_features, seq_target = self.sequences[i][0].copy(), self.sequences[i][1].copy()

            seq_features[:, :5] = (seq_features[:, :5] - self.mean[:, :5]) / self.std[:, :5]
            seq_target = (seq_target - self.t_mean) / self.t_std

            self.sequences[i] = (seq_features, seq_target)

    def _create_sequences(self):
        sequences = []
        for i in range(len(self.features) - self.sequence_length):
            seq_features = self.features[i:i + self.sequence_length].copy()
            seq_target = self.targets[i + self.sequence_length].copy()

            # Check time gaps
            seq_timestamps = self.timestamps[i:i + self.sequence_length]
            time_gaps = (seq_timestamps[1:] - seq_timestamps[:-1]) / np.timedelta64(1, 's')  # Convert to seconds

            if (time_gaps >= 1.1).any():
                continue

            seq_features[:, :2] = seq_features[0, :2]
            
            # Rearrange columns: move last two to front, move first two after speed
            cols = list(range(seq_features.shape[1]))
            new_order = [-2, -1, 2] + [0, 1] + cols[3:-2]  # [n-2, n-1, 2, 0, 1, 3, 4, ..., n-3]
            seq_features = seq_features[:, new_order]

            sequences.append((seq_features, seq_target))
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {'input': torch.tensor(self.sequences[idx][0], dtype=torch.float32), 
                'label': torch.tensor(self.sequences[idx][1], dtype=torch.float32)}