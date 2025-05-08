import os, torch
from torch.utils.data import DataLoader
from dataset_loader import ArgoverseDataset, TrajectoryPredictionDataset, RoadPredictionDataset
# from download_pvs import download_and_process_data
from download_argoverse import check_and_download_argoverse

def load_super(inputs, dataset='Argoverse'):
    if dataset == 'Argoverse':
        check_and_download_argoverse()
        train_loader, val_loader, test_loader = setup_argoverse_loaders(*inputs)
    elif dataset == 'Trajectory':
        # download_and_process_data()
        train_loader, val_loader, test_loader = setup_pvs_loaders('trajectory', *inputs)
    elif dataset == 'Road':
        # download_and_process_data()
        train_loader, val_loader, test_loader = setup_pvs_loaders('road', *inputs)

    return train_loader, val_loader, test_loader

def load_legacy_data(file_path, Loader, seq_len):
    file_paths = []
    for i in range(1, 10):
        file_paths.append(os.path.join(file_path, f'{i}.json'))

    return Loader(file_paths, seq_len, f'cached_{seq_len}_{file_path}')

def setup_argoverse_loaders(batch_size, num_agents):
    train_dataset = ArgoverseDataset('train', num_agents=num_agents)
    val_dataset = ArgoverseDataset('val', num_agents=num_agents)
    test_dataset = ArgoverseDataset('test', num_agents=num_agents)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def setup_pvs_loaders(choice, batch_size, seq_len, testing_split=0.0, validation_split=0.2):
    if choice == 'trajectory':
        dataset = load_legacy_data('trajectory_prediction', TrajectoryPredictionDataset, seq_len)
    elif choice == 'road':
        dataset = load_legacy_data('road_prediction', RoadPredictionDataset, seq_len)
    else:
        raise ValueError(f'The provided choice {choice} is invalid.')

    test_size = int(testing_split * len(dataset))
    val_size = int(validation_split * (len(dataset) - test_size))
    train_size = len(dataset) - test_size - val_size

    dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size + val_size, test_size])
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader