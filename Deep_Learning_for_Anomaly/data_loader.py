import os
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import normalize, MultiLabelBinarizer, MinMaxScaler, LabelEncoder
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

class DataLoader:
    """
    DataLoader class for handling fault detection and prognostics data
    Supports both detection (current state) and prognostics (future state) tasks
    """
    def __init__(self, data_path='results/data/module_RR.pkl', task='detection'):
        """
        Initialize DataLoader
        Args:
            data_path: path to data file
            task: 'detection' or 'prognostics'
        """
        self.data_path = data_path
        self.task = task
        assert task in ['detection', 'prognostics'], "Task must be either 'detection' or 'prognostics'"
        
        # Create data directory if it doesn't exist
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """
        Load data from pickle file
        Returns:
            tuple: (data, labels, readme information)
        """
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Data loaded with length: {len(data)}")
        return data['data'], data['label'], data['README']
    
    def fault_check(self, labels, save_path=None):
        """
        Check fault distribution and optionally save results
        Args:
            labels: Array of fault labels
            save_path: Directory to save visualization results
        Returns:
            DataFrame: Fault distribution statistics
        """
        # Calculate fault distribution
        unique_fault, counts = np.unique(labels, return_counts=True)
        df = pd.DataFrame({'event':unique_fault, 'count':counts}).drop(0).sort_values(by=['count'])
        
        if save_path:
            # Create visualization directory
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot and save fault distribution
            plt.figure(figsize=(8, 3), dpi=300)
            df["event"] = df["event"].astype(int).astype(str)
            df.to_csv(save_dir / 'fault_counts.csv', index=False)
            plt.bar(df['event'].to_list(), df['count'].to_list(), color='maroon', width=0.4)
            plt.title('Counts of All Faults in the System')
            plt.xlabel("Fault Type")
            plt.ylabel("Count")
            plt.savefig(save_dir / 'fault_distribution.png', bbox_inches='tight')
            plt.close()
            
        return df
    
    def preprocess_labels(self, Y):
        """
        Filter and preprocess labels to keep only important faults
        Args:
            Y: Raw fault labels
        Returns:
            tuple: (processed labels, label encoder)
        """
        # Define important fault codes
        useful_code = [0.0, 55.0, 67.0, 95.0]  # Normal operation and 3 fault types
        unique_fault = np.unique(Y).tolist()
        rare_list = [x for x in unique_fault if x not in useful_code]
        rare_list = np.array(rare_list)
        
        # Group rare faults together
        processed_Y = Y.copy()
        for rare_value in rare_list:
            processed_Y[processed_Y == rare_value] = 100
            
        # Encode labels
        le = LabelEncoder()
        shape_Y = processed_Y.shape
        Y_le = le.fit_transform(processed_Y.flatten('C'))
        Y_le = Y_le.reshape(shape_Y, order='C')
        
        return Y_le, le
    
    def sliding_windows(self, X, Y, stride=30, sequence_length=100, forecast_length=100):
        """
        Create sliding windows from data for sequence processing
        Args:
            X: Input features
            Y: Labels
            stride: Window stride
            sequence_length: Length of input sequence
            forecast_length: Length of future sequence (for prognostics)
        Returns:
            tuple: (windowed features, current labels, future labels)
        """
        x, y, forecast_y = [], [], []
        for i in range(int((len(X)-sequence_length)/stride)+1-stride):
            _x = X[i*stride:i*stride+sequence_length,...]
            _y = Y[i*stride:i*stride+sequence_length,...]
            if i <= (int((len(X)-sequence_length)/stride)+1)-stride-1:
                _f = Y[i*stride+sequence_length:i*stride+sequence_length+forecast_length,...]
                forecast_y.append(_f)
            x.append(_x)
            y.append(_y)
        return np.array(x), np.array(y), np.array(forecast_y)
    
    def convert_to_multilabel(self, labels, future_labels=None):
        """
        Convert labels to multilabel format
        Args:
            labels: Current state labels
            future_labels: Future state labels (for prognostics)
        Returns:
            array or tuple: Multilabel format labels
        """
        mlb = MultiLabelBinarizer()
        y_multilabel = mlb.fit_transform(labels)
        y_multi = y_multilabel[:,1:]  # Remove normal operation column
        
        if future_labels is not None:
            y_futurelabel = mlb.transform(future_labels)
            y_future = y_futurelabel[:,1:]
            return y_multi, y_future
        return y_multi
    
    def prepare_data(self, device='cuda:3', batch_size=40):
        """
        Main method to prepare data for training
        Args:
            device: Computing device (CPU/GPU)
            batch_size: Batch size for training
        Returns:
            tuple: (train_loader, test_loader, label_encoder)
        """
        # Load and preprocess data
        X, Y, _ = self.load_data()
        Y_le, le = self.preprocess_labels(Y)
        
        # Reshape and normalize features
        Xshape = X.shape
        X_reshape = X.reshape((Xshape[0]*Xshape[1],Xshape[2],Xshape[3]))
        X_reshape = np.swapaxes(X_reshape,2,0)
        X_reshape = np.swapaxes(X_reshape,2,1)[...,:-1]
        
        # Normalize features
        Min_max = MinMaxScaler()
        X_norm = X_reshape.reshape(-1,1)
        X_norm = Min_max.fit_transform(X_norm)
        X_norm = X_norm.reshape(X_reshape.shape)
        
        # Split data into train and test sets
        train_len = int(np.floor(len(Y_le)/10*0.8)*10)-60
        test_len = int(np.floor(len(Y_le)/10*0.2)*10)-40
        
        train_X = X_norm[:train_len,:]
        train_Y = Y_le[:train_len]
        test_X = X_norm[train_len:train_len+test_len,:]
        test_Y = Y_le[train_len:train_len+test_len]
        
        # Create windows
        train_X_win, train_Y_win, train_f_y = self.sliding_windows(train_X, train_Y)
        test_X_win, test_Y_win, test_f_y = self.sliding_windows(test_X, test_Y)
        
        # Reshape for all modules
        train_X = np.swapaxes(train_X_win,1,2).reshape((-1, train_X_win.shape[2], train_X_win.shape[3]))
        test_X = np.swapaxes(test_X_win,1,2).reshape((-1, test_X_win.shape[2], test_X_win.shape[3]))
        
        # Prepare labels based on task
        if self.task == 'detection':
            train_Y = np.swapaxes(train_Y_win,1,2).reshape((-1, train_Y_win.shape[2]))
            test_Y = np.swapaxes(test_Y_win,1,2).reshape((-1, test_Y_win.shape[2]))
            train_labels = self.convert_to_multilabel(train_Y)
            test_labels = self.convert_to_multilabel(test_Y)
        else:  # prognostics
            train_Y = np.swapaxes(train_Y_win,1,2).reshape((-1, train_Y_win.shape[2]))
            train_f_y = np.swapaxes(train_f_y,1,2).reshape((-1, train_f_y.shape[2]))
            test_Y = np.swapaxes(test_Y_win,1,2).reshape((-1, test_Y_win.shape[2]))
            test_f_y = np.swapaxes(test_f_y,1,2).reshape((-1, test_f_y.shape[2]))
            train_labels, train_future = self.convert_to_multilabel(train_Y, train_f_y)
            test_labels, test_future = self.convert_to_multilabel(test_Y, test_f_y)
        
        # Create DataLoaders
        if self.task == 'detection':
            train_data = torch.utils.data.TensorDataset(
                torch.Tensor(train_X).float().to(device),
                torch.Tensor(train_labels).float().to(device)
            )
            test_data = torch.utils.data.TensorDataset(
                torch.Tensor(test_X).float().to(device),
                torch.Tensor(test_labels).float().to(device)
            )
        else:
            train_data = torch.utils.data.TensorDataset(
                torch.Tensor(train_X).float().to(device),
                torch.Tensor(train_labels).float().to(device),
                torch.Tensor(train_future).float().to(device)
            )
            test_data = torch.utils.data.TensorDataset(
                torch.Tensor(test_X).float().to(device),
                torch.Tensor(test_labels).float().to(device),
                torch.Tensor(test_future).float().to(device)
            )
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=10000)
        
        return train_loader, test_loader, le

if __name__ == "__main__":
    # Create results directory structure
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Example usage for both tasks
    for task in ['detection', 'prognostics']:
        print(f"\nPreparing data for {task}...")
        data_loader = DataLoader(task=task)
        train_loader, test_loader, le = data_loader.prepare_data()
        
        # Save fault distribution analysis
        X, Y, _ = data_loader.load_data()
        save_path = results_dir / task / "data_analysis"
        data_loader.fault_check(Y, save_path=save_path)
        
        print(f"Data preparation for {task} completed!")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of test batches: {len(test_loader)}") 