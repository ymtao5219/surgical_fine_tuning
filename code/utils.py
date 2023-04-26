import numpy as np
import json
import matplotlib.pyplot as plt
import random
import pickle 

def load_dict_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        dict_obj = pickle.load(f)
    return dict_obj

def save_dict_to_pickle(dict_obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dict_obj, f)

def save_numpy_array_to_file(array, file_path):
    np.save(file_path, array)
    
def load_numpy_array_from_file(file_path):
    return np.load(file_path)

def plot_fim_per_layer(density, file_name):
    pass 
