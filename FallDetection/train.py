import scipy.io
import h5py
import numpy as np

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils import *
from models import *
# device = torch.device('cpu')

base_dir = './denoisemat/fall/'  # e.g., '/home/user/experiment'
file_list = get_sorted_mat_files(base_dir)
csi_array_fall = aggregate_mat_files(file_list, var_name='csi_data')
activity_array_fall = aggregate_mat_files(file_list, var_name='activity')
user_array_fall = aggregate_mat_files(file_list, var_name='user')

base_dir = './denoisemat/nonfall/'  # e.g., '/home/user/experiment'
file_list = get_sorted_mat_files(base_dir)
csi_array_nonfall = aggregate_mat_files(file_list, var_name='csi_data')
activity_array_nonfall = aggregate_mat_files(file_list, var_name='activity')
user_array_nonfall = aggregate_mat_files(file_list, var_name='user')

csi_fall = torch_data(csi_array_fall)
csi_nonfall = torch_data(csi_array_nonfall)


labels0 = torch.zeros(csi_fall.shape[0], dtype=torch.long)    # All zeros
labels1 = torch.ones(csi_nonfall.shape[0], dtype=torch.long)     # All ones

# Concatenate data and labels
csi = torch.cat([csi_fall, csi_nonfall], dim=0)
label = torch.cat([labels0, labels1], dim=0)

N = csi.shape[0]
torch.manual_seed(42)
perm = torch.randperm(N)

csi_shuffled   = csi[perm]
label_shuffled = label[perm]

size = csi.shape[0]
portion = 0.7
dataset_train = TensorDataset(csi_shuffled[:int(portion * size)], label_shuffled[:int(portion*size)])
dataset_test = TensorDataset(csi_shuffled[int(portion * size):], label_shuffled[int(portion * size):])
loader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)  # Set shuffle=True here
loader_test = DataLoader(dataset_test, batch_size=16, shuffle=True)  # Set shuffle=True here

from tqdm import tqdm
import os
num_epochs = 10

train_model(
    model_dir='./model/Transformer/',
    model = FD_TransformerE(),
    loader_train=loader_train,
    epoch = 10,
)




