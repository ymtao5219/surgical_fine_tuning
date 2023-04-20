import numpy as np
import json
import matplotlib.pyplot as plt
import random
import csv

def sample_probe_set(file_name, num_of_sentences, seed):
    random.seed(seed)
    with open(file_name, 'r', newline='') as f:
        reader = csv.reader(f)
        # Extract the attribute name and sentences
        attribute_name = next(reader)[0].strip()
        sentences = [row[0].strip() for row in reader]
        # Sample num_of_sentences sentences randomly
        sample = random.sample(sentences, k=num_of_sentences)
        return attribute_name, sample

def sample_negative_sentences(file_name, M, N, seed=42):
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        sentences = [row[0] for row in reader]

    samples = []
    for i in range(M):
        random.seed(seed + i)
        sample = random.sample(sentences, N)
        samples.append(sample)

    return samples

def save_numpy_array_to_file(array, file_path):
    np.save(file_path, array)
    
def load_numpy_array_from_file(file_path):
    return np.load(file_path)

def plot_density_per_layer(density, file_name):
    pass 

def plot_heatmap(binary_arrays, file_name):
    # Stack the binary arrays vertically to form a 2D NumPy array
    nrows, ncols = binary_arrays.shape
    
    # create a figure and subplot for the heatmap
    fig, ax = plt.subplots()
    heatmap = ax.imshow(binary_arrays, cmap='viridis', aspect=12)
    ax.set_title("Heatmap of activated neurons")
    plt.colorbar(heatmap)
    ax.set_xlabel("Neurons")
    ax.set_ylabel("Layers")
    fig.savefig("./figs/" + file_name + ".png", dpi=500, bbox_inches='tight')