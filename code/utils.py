import numpy as np
import json
import matplotlib.pyplot as plt
import random
import pickle 
import torch

def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dict_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        dict_obj = pickle.load(f)
    return dict_obj

def save_dict_to_pickle(dict_obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dict_obj, f)

def save_numpy_array_to_file(array, file_path):
    np.save(file_path, array)
