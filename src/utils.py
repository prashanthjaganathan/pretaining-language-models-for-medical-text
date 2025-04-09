import h5py
import numpy as np
import pickle

def save_embeddings_to_file(embeddings, file_path):
    """
    Saves embeddings to an HDF5 file.

    Args:
        embeddings (list or numpy array): The embeddings to store.
        file_path (str): The HDF5 file path.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(embeddings, file)
        
    print(f"Embeddings saved to {file_path}")

