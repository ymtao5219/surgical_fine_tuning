import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt

def save_numpy_array_to_file(array, file_path):
    np.save(file_path, array)
    
def load_numpy_array_from_file(file_path):
    return np.load(file_path)

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

import seaborn as sns

def plot_heatmap(binary_arrays, file_path="./figs/heatmap.png"):

    # Stack the binary arrays vertically to form a 2D NumPy array
    stacked_arrays = np.vstack(binary_arrays)

    nrows, ncols = stacked_arrays.shape
    # plot the binary matrix as an image
    plt.imshow(binary_arrays, cmap='binary', aspect=12)
    plt.colorbar()
    
    if file_path:
        plt.savefig(file_path, dpi=500, bbox_inches='tight')
