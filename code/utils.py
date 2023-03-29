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

def plot_bipartite_graph(my_dict):
    G = nx.Graph()
    top_nodes = list(my_dict.keys())
    bottom_nodes = list(set([val for sublist in my_dict.values() for val in sublist]))
    G.add_nodes_from(top_nodes, bipartite=0)
    G.add_nodes_from(bottom_nodes, bipartite=1)
    for key, values in my_dict.items():
        for value in values:
            G.add_edge(key, value)
    pos = {node: (0, i) for i, node in enumerate(top_nodes)}
    pos.update({node: (1, i) for i, node in enumerate(bottom_nodes)})
    nx.draw(G, pos=pos, with_labels=True, node_color=["red"]*len(top_nodes) + ["green"]*len(bottom_nodes), node_shape="s")
    plt.savefig("./figs/bipartite_graph.png")
    plt.show()
