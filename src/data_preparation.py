import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def prepare_data(test_size=0.2, val_size=0.2, random_state=42):
    """
    Loads and prepares the MNIST dataset.
    
    Parameters:
    - test_size (float): Proportion of the dataset to include in the test split.
    - val_size (float): Proportion of the train dataset to include in the validation split.
    - random_state (int): Random seed for reproducibility.
    
    Returns:
    - x_train, x_val, x_test (np.ndarray): Training, validation, and testing feature sets.
    - y_train, y_val, y_test (np.ndarray): Corresponding labels.
    """
    mnist = fetch_openml('mnist_784', as_frame=False, cache=True)
    x = mnist["data"]
    y = mnist["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y, shuffle=True
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train, shuffle=True
    )
    return x_train, x_val, x_test, y_train, y_val, y_test

def convert_to_tensor(data, dtype=torch.float32, reshape=None, normalize=False):
    """
    Convert a NumPy array to a PyTorch tensor with optional reshaping and normalization.
    
    Parameters:
    - data (np.ndarray): Input NumPy array.
    - dtype (torch.dtype): Desired PyTorch data type.
    - reshape (tuple): New shape for the tensor (optional).
    - normalize (bool): Whether to normalize data to [0, 1].
    
    Returns:
    - torch.Tensor: Converted PyTorch tensor.
    """
    tensor = torch.tensor(data, dtype=dtype)
    if reshape:
        tensor = tensor.reshape(reshape)
    if normalize:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return tensor

def prepare_tensors(x_train, x_val, x_test, y_train, y_val, y_test):
    """
    Prepares PyTorch tensors for the dataset by reshaping and normalizing.

    Parameters:
    - x_train, x_val, x_test (np.ndarray): Feature datasets.
    - y_train, y_val, y_test (np.ndarray): Label datasets.

    Returns:
    - Tensors for training, validation, and testing datasets (features and labels).
    """
    x_train_tensor = convert_to_tensor(x_train, dtype=torch.float32, reshape=(-1, 1, 28, 28), normalize=True)
    x_val_tensor = convert_to_tensor(x_val, dtype=torch.float32, reshape=(-1, 1, 28, 28), normalize=True)
    x_test_tensor = convert_to_tensor(x_test, dtype=torch.float32, reshape=(-1, 1, 28, 28), normalize=True)

    y_train_tensor = convert_to_tensor(y_train.astype(int), dtype=torch.long)
    y_val_tensor = convert_to_tensor(y_val.astype(int), dtype=torch.long)
    y_test_tensor = convert_to_tensor(y_test.astype(int), dtype=torch.long)
    
    return x_train_tensor, x_val_tensor, x_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor
