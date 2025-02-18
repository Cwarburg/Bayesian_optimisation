import os
import pickle
import numpy as np
from sklearn.datasets import fetch_openml

def load_data(cache_path="data/mnist_data.pkl", force_reload=False):
    """
    Loads the MNIST dataset, preprocesses it, and caches it to disk.
    
    Parameters:
        cache_path (str): File path to save or load cached data. Defaults to 'data/mnist_data.pkl'.
        force_reload (bool): If True, ignores any cached file and downloads data anew.
        
    Returns:
        X (np.ndarray): Normalized pixel data of shape (n_samples, 784)
        y (np.ndarray): Digit labels as integers.
    """
    # Ensure the directory for the cache exists
    cache_dir = os.path.dirname(cache_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Created directory '{cache_dir}' for cached data.")
    
    # Load from cache if available and not forcing a reload
    if os.path.exists(cache_path) and not force_reload:
        print(f"Loading cached data from '{cache_path}'...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return data["X"], data["y"]
    
    # Download and preprocess the data
    print("Downloading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    
    # Convert data types for consistency
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int32)
    
    # Normalize pixel values to the range [0, 1]
    X /= 255.0
    
    # Save the preprocessed data to the specified cache path
    data = {"X": X, "y": y}
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Data cached at '{cache_path}'.")
    return X, y

if __name__ == "__main__":
    X, y = load_data()
    print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")
