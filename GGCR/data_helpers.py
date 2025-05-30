# for GPU programming
import os
import logging
import torch
import numpy as np
import pandas as pd

#logger function for logging information
def logger_fn(name, input_file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(input_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger

#loading data
def load_data(input_file, flag=None):
    if flag:
        data = pd.read_json(input_file, orient='records', lines=True)
        #data = pd.read_csv(input_file, orient='records', lines=True)
    else:
        data = pd.read_json(input_file, orient='records', lines=True)
        #data = pd.read_csv(input_file, orient='records', lines=True)

    return data

#loading model file
def load_model_file(checkpoint_dir):
    MODEL_DIR = './runs_v20/' + checkpoint_dir
    
    names = [name for name in os.listdir(MODEL_DIR) if os.path.isfile(os.path.join(MODEL_DIR, name))]
    max_epoch = 0
    choose_model = ''
    for name in names:
        if int(name[6:8]) >= max_epoch:
            max_epoch = int(name[6:8])
            choose_model = name
    MODEL_FILE = './runs_v20/' + checkpoint_dir + '/' + choose_model
    #MODEL_FILE = checkpoint_dir + '/' + choose_model
    return MODEL_FILE

def sort_batch_of_lists(uids, batch_of_lists, lens, device):
    """Sort batch of lists according to len(list). Descending"""
    sorted_idx = [i[0] for i in sorted(enumerate(lens), key=lambda x: x[1], reverse=True)]
    uids = [uids[i] for i in sorted_idx]
    lens = [lens[i] for i in sorted_idx]
    batch_of_lists = [batch_of_lists[i] for i in sorted_idx]
    prev_idx = []
    for idx in sorted_idx:
       prev_idx.append(1)
    return uids, batch_of_lists, lens, prev_idx

def sort_batch_of_lists_2(uids, batch_of_lists, lens, last_batch_actual_size, device):
    """Sort batch of lists according to len(list). Descending"""
    sorted_idx = [i[0] for i in sorted(enumerate(lens), key=lambda x: x[1], reverse=True)]
    uids = [uids[i] for i in sorted_idx]
    lens = [lens[i] for i in sorted_idx]
    batch_of_lists = [batch_of_lists[i] for i in sorted_idx]
    prev_idx = []
    for idx in sorted_idx:
        if(idx<last_batch_actual_size):
            prev_idx.append(1)
        else:
            prev_idx.append(0) #randomly taken to maintain fixed length batch size

    return uids, batch_of_lists, lens, prev_idx

def pad_batch_of_lists(batch_of_lists, pad_len, device):
    """Pad batch of lists."""
    padded = [l + [[0]] * (pad_len - len(l)) for l in batch_of_lists]
    return padded
def pad_batch_of_lists2(batch_of_lists, pad_len, device):
    """Pad batch of lists."""
    padded = [l + [0] * (pad_len - len(l)) for l in batch_of_lists]
    padded = torch.tensor(padded, device = device)
    return padded

def batch_iter(data, batch_size, pad_len, device, shuffle=True, seed_value=42):
    """
    Turn dataset into iterable batches with reproducibility.

    Args:
        data: The data
        batch_size: The size of the data batch
        pad_len: The padding length
        device: The computation device (CPU or GPU)
        shuffle: Shuffle the data or not (default: True)
        seed_value: Random seed for reproducibility
    Returns:
        A batch iterator for the dataset
    """
    data_size = len(data)
    num_batches_per_epoch = data_size // batch_size

    # Set seed for reproducibility
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if shuffle:
        shuffled_data = data.sample(frac=1, random_state=seed_value)  # Ensure deterministic shuffle
    else:
        shuffled_data = data

    generator = torch.Generator().manual_seed(seed_value)  # PyTorch random generator

    for i in range(num_batches_per_epoch):
        uids = torch.tensor(
            shuffled_data.iloc[i * batch_size: (i + 1) * batch_size].userID.values, 
            device=device
        )
        baskets = list(shuffled_data.iloc[i * batch_size: (i + 1) * batch_size].baskets.values)
        lens = torch.tensor(
            shuffled_data.iloc[i * batch_size: (i + 1) * batch_size].num_baskets.values, 
            device=device
        )

        uids, baskets, lens, prev_idx = sort_batch_of_lists(uids, baskets, lens, device)  
        baskets = pad_batch_of_lists(baskets, pad_len, device)
        yield uids, baskets, lens, prev_idx

    if data_size % batch_size != 0:
        np.random.seed(seed_value)  # Reset seed for reproducibility
        torch.manual_seed(seed_value)

        residual = [i for i in range(num_batches_per_epoch * batch_size, data_size)] + list(
            np.random.choice(data_size, batch_size - data_size % batch_size, replace=True)
        )

        uids = torch.tensor(
            shuffled_data.iloc[residual].userID.values, 
            device=device   
        )
        baskets = list(shuffled_data.iloc[residual].baskets.values)
        lens = torch.tensor(
            shuffled_data.iloc[residual].num_baskets.values, 
            device=device
        )

        uids, baskets, lens, prev_idx = sort_batch_of_lists_2(uids, baskets, lens, data_size % batch_size, device)
        baskets = pad_batch_of_lists(baskets, pad_len, device)
        yield uids, baskets, lens, prev_idx

def pool_max(tensor, dim):
    return torch.max(tensor, dim)[0]

def pool_avg(tensor, dim):
    return torch.mean(tensor, dim)
