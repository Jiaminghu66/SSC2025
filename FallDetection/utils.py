import scipy.io
import h5py
import numpy as np

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

def torch_data(X_complex):
    """
    Convert [batch, 2000, 30, 3] complex NumPy array
    to [batch, 2000, 90, 2] real NumPy array (real, imag split).
    """
    batch, t, ch, three = X_complex.shape
    assert three == 3, "Last dimension must be 3 (3 complex signals per channel)."

    # Reshape: [batch, 2000, 30, 3] → [batch, 2000, 90]
    X_flat = X_complex.reshape(batch, t, ch * three)

    # Get real and imag parts
    real = np.real(X_flat)
    imag = np.imag(X_flat)

    # Stack along the last axis to get [batch, 2000, 90, 2]
    X_realimag = torch.tensor(
        np.stack([real, imag], axis=-1),
        dtype = torch.float32
    )
    return X_realimag


def read_mat(filename):
    """
    Load a .mat file (MATLAB format) and return a Python dictionary of variables.
    Supports both v7/v6 and v7.3 (HDF5) formats.

    Args:
        filename (str): Path to the .mat file.

    Returns:
        dict: A dictionary containing the variables from the .mat file.
    """
    try:
        # Try scipy.io for v7/v6 files
        mat = scipy.io.loadmat(filename)
        # Remove meta entries
        mat = {k: v for k, v in mat.items() if not k.startswith('__')}
        return mat
    except NotImplementedError:
        # If file is v7.3 (HDF5), use h5py
        print("scipy.io.loadmat failed, trying h5py for HDF5 v7.3 file...")
        with h5py.File(filename, 'r') as f:
            def hdf5_to_dict(hdf5_obj):
                out = {}
                for key, item in hdf5_obj.items():
                    if isinstance(item, h5py.Dataset):
                        out[key] = item[()]
                    elif isinstance(item, h5py.Group):
                        out[key] = hdf5_to_dict(item)
                return out
            return hdf5_to_dict(f)

def read_dat(filename, dtype=np.float32, shape=None):
    """
    Read a binary .dat file containing raw float or integer data.

    Args:
        filename (str): Path to the .dat file.
        dtype (numpy dtype): Data type (np.float32, np.float64, etc.)
        shape (tuple or None): Desired array shape. If None, returns flat array.

    Returns:
        np.ndarray: Array of data.
    """
    data = np.fromfile(filename, dtype=dtype)
    if shape is not None:
        data = data.reshape(shape)
    return data

import os
import glob

def get_sorted_mat_files(base_dir, folder_pattern='20*', file_pattern='*.mat'):
    """
    Get all .mat files from subfolders in base_dir, sorted by folder and filename.
    Returns a sorted list of (folder, filename) tuples.
    """
    folders = sorted(glob.glob(os.path.join(base_dir, folder_pattern)))
    all_files = []
    for folder in folders:
        mat_files = sorted(glob.glob(os.path.join(folder, file_pattern)))
        all_files.extend(mat_files)
    return all_files


def aggregate_mat_files(file_list, var_name='data'):
    """
    Reads all .mat files in file_list and stacks their 'data' variables into a single array.
    Assumes all data arrays have the same shape.
    """
    all_data = []
    for fname in file_list:
        mat = read_mat(fname)
        if var_name in mat:
            data = mat[var_name]
            # Squeeze singleton dims for consistency
            if hasattr(data, 'squeeze'):
                data = data.squeeze()
            all_data.append(data)
        else:
            print(f"Warning: {fname} does not contain variable '{var_name}'")
    # Stack along new axis (e.g., 0)
    total_array = np.stack(all_data, axis=0)
    return total_array

def save_model(model, model_dir, model_name,max_keep=1):
  # Create the directory if it doesn't exist
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
 
  # Save the model
  model_path = os.path.join(model_dir, model_name)
  torch.save(model.state_dict(), model_path)
 
  # Get all saved models and sort them by creation time
  saved_models = sorted(os.listdir(model_dir), key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
 
  # If there are more than 3 models, delete the oldest ones
  if len(saved_models) > max_keep:
    # print('del:',saved_models[0])
    os.remove(os.path.join(model_dir, saved_models[0]))
  return 0

def train_model(model_dir, model, loader_train, epoch = 10, device = torch.device('cuda')):
    # model_dir = './model/CNN_GRU/'
    best_loss = 10
    # model = FD_CNNGRU()
    if device == torch.device('cuda'):
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _epoch in range(epoch):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        train_bar = tqdm(loader_train, desc=f"Epoch {_epoch+1}/{epoch} [Train]", leave=False)
        for xb, yb in train_bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            pred = torch.argmax(logits, dim=1)
            acc = (pred == yb).sum().item() / xb.size(0)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
            train_bar.set_postfix(loss=loss.item())
            if loss < best_loss:
                best_loss = loss
                save_model(model, model_dir, f'{acc*100:.0f}.pth')

        train_acc = correct / total
        avg_loss = total_loss / total
        print(f"Epoch {epoch+1}: Train loss = {avg_loss:.4f}, accuracy = {train_acc:.3f}")

def test_model(model, model_dir, loader_test):
    device = torch.device('cuda')
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    state_dict = torch.load(model_dir, weights_only = True)
    model.load_state_dict(state_dict)
    model.eval()
    correct = 0
    total = 0

    test_bar = tqdm(loader_test, desc=f"[Test]", leave=False)
    for xb, yb in test_bar:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)

    train_acc = correct / total
    print(f"Test accuracy = {train_acc:.3f}")