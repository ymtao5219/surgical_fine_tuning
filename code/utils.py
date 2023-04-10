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

def plot_heatmap(binary_arrays, file_name):
    # Stack the binary arrays vertically to form a 2D NumPy array
    stacked_arrays = np.vstack(binary_arrays)

    nrows, ncols = stacked_arrays.shape
    
    # create a figure and subplot for the heatmap
    fig, ax = plt.subplots()
    heatmap = ax.imshow(binary_arrays, cmap='binary', aspect=12)
    ax.set_title("Heatmap of activated neurons")
    plt.colorbar(heatmap)
    ax.set_xlabel("Neurons")
    ax.set_ylabel("Layers")
    fig.savefig("./figs/" + file_name + ".png", dpi=500, bbox_inches='tight')