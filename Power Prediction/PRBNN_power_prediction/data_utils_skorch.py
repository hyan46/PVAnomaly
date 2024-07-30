import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import copy
def load_data(file_path="Know_regression.pkl"):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def preprocess_data(data):
    # Reshaping and preprocessing as per your instructions
    temp = np.swapaxes(data, 0, 1)
    P_W_data = np.swapaxes(temp, 1, 2)
    com_P_W_data = np.reshape(P_W_data, (731*96, 12))

    # Adding time and date labels
    time_label = torch.ones(70176).unsqueeze(1).numpy()
    for i in range(731):
        time_label[96*i:96*(i+1)] = np.arange(96).reshape((96, 1))

    days = np.concatenate((np.arange(365), np.arange(366)))
    n_days = np.repeat(days, 96)[:, np.newaxis]

    com_P_W_data = np.concatenate((com_P_W_data, time_label, n_days), axis=1)

    # Monthly label
    months = np.concatenate([np.arange(12), np.arange(12)])
    month_label = months.repeat(30*96)
    month_label = np.pad(month_label, (0, len(com_P_W_data)-len(month_label)), 'constant')[:, np.newaxis]
    com_P_W_data = np.concatenate((com_P_W_data, month_label), axis=1)

    # Select specific features for P_W_data_inverterA
    # Assuming the order and the inclusion of time, day, and month labels are correct, adjust indices as needed
    feature_indices = [0, 1, 2, 5, 7, 8, 9, 10, 11, 12, 13] 
    P_W_data_inverterA = com_P_W_data[:, feature_indices]

    return P_W_data_inverterA

def normalize_data(data):
    for d in range(data.shape[-1]):
        data[..., d] = (data[..., d] - np.min(data[..., d])) / (np.max(data[..., d]) - np.min(data[..., d]) + 1e-6)
    X = data[..., 1:]
    Y = data[..., 0]
    X_torch = torch.from_numpy(X).float()
    Y_torch = torch.from_numpy(Y).float()
    return X_torch, Y_torch

def split_data(X,Y, train_ratio=0.7):
    # Assuming the last dimension is the feature dimension
    train_idx = round(train_ratio * len(Y))
    y_train = Y[:train_idx]
    y_test = Y[train_idx:]
    x_train = X[:train_idx]
    x_test = X[train_idx:]
    return x_train, x_test, y_train, y_test



def derate_dataloader(device):
    derate_series=torch.tensor([[0.0, 0.0000, 0.0000, 0.4030,  1.0000, 0.0000, 1.0000,1, 0.3, 0.26],
                            [0.0, 0.0000, 0.0000, 0.95,     1.0000, 0.0000, 1.0000,1, 0.5, 0.25],
                            [0.0, 0.0000, 0.0000, 0.6030,    1.0000, 0.0000, 1.0000,1, 0.652, 0.9808],
                            [0.0, 0.0000, 0.0000, 0.7030,   1.0000, 0.0000, 1.0000,1, 0.4842, 0.2137],
                            [0.0, 0.0000, 0.0000, 0.89,     1.0000, 0.0000, 1.0000,1, 1.5895, 0.9808],
                            [0.0, 0.0000, 0.0000, 0.9278, 1.0000, 0.0000, 1.0000, 1., 0.6211, 0.0055],
                            [0.0, 0.0000, 0.0000, 0.4030,  0.0000, 0.0000, 1.0000,1, 0.3, 0.26],
                            [0.0, 0.0000, 0.0000, 0.95,      0.0000, 0.0000, 1.0000,1, 0.5, 0.25],
                            [0.0, 0.0000, 0.0000, 0.6030,    0.0000, 0.0000, 1.0000,1, 0.652, 0.9808],
                            [0.0, 0.0000, 0.0000, 0.7030,   0.0000, 0.0000, 1.0000,1, 0.4842, 0.2137],
                            [0.0, 0.0000, 0.0000, 0.89,     0.0000, 0.0000, 1.0000,1, 0.5895, 0.9808],
                            [0.0, 0.0000, 0.0000, 0.9278, 0.0000, 0.0000, 1.0000, 1., 0.6211, 0.0055]]
                             ).to(device)
    
    
    return derate_series
    