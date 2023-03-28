import numpy as np

"""
This file contains helper functions. 
"""

def save_numpy_array_to_file(array, file_path):
    np.save(file_path, array)
    
def load_numpy_array_from_file(file_path):
    return np.load(file_path)

