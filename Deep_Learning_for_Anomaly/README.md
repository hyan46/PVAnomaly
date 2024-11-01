# Deep Learning for Anomaly Detection and Prognostics

This project implements a deep learning approach for both fault detection and prognostics in industrial systems. It uses a Convolutional Neural Network (CNN) architecture to analyze time-series data and identify/predict system faults.

## Project Structure

```
Deep_Learning_for_Anomaly/
├── data_loader.py      # Data processing and loading utilities
├── model.py            # CNN model and training components
├── results/            # Generated results and visualizations
│   ├── data/          # Raw and processed data
│   ├── detection/     # Fault detection results
│   │   ├── data_analysis/
│   │   ├── images/    # Confusion matrices and plots
│   │   └── models/    # Saved model weights
│   └── prognostics/   # Fault prognostics results
│       ├── data_analysis/
│       ├── images/    # Confusion matrices and plots
│       └── models/    # Saved model weights
└── README.md          # This file
```

## Features

- **Dual-Task Support**: Handles both fault detection and prognostics
- **Data Processing**: Includes sliding window generation and multi-label conversion
- **Model Architecture**: CNN-based architecture with task-specific output layers
- **Comprehensive Evaluation**: F1 scores and confusion matrices for each fault type
- **Visualization**: Automated generation of performance plots and fault distributions

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

Install requirements:
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

## Usage

1. **Data Preparation**:
   - Place your data file (module_RR.pkl) in the `results/data/` directory
   - The data should contain:
     - Time-series measurements
     - Fault labels
     - Optional README information

2. **Running the Code**:
   ```bash
   # For data processing only
   python data_loader.py
   
   # For model training and evaluation
   python model.py
   ```

3. **Configuration**:
   - Adjust hyperparameters in the respective files:
     - Batch size
     - Learning rate
     - Number of epochs
     - Model architecture
     - Window sizes

## Model Architecture

The CNN model consists of:
- Two convolutional blocks with max pooling
- Task-specific output layers:
  - Detection: Single output layer for current state
  - Prognostics: Dual output layers for current and future states

### Key Parameters:
- Input channels: 1
- First conv layer: 64 filters, 5x2 kernel
- Second conv layer: 32 filters, 5x2 kernel
- Output: 4 fault types

## Data Processing

The data loader handles:
- Loading and preprocessing raw data
- Creating sliding windows for sequence analysis
- Converting to multi-label format
- Train/test splitting
- Data normalization
- Batch preparation

## Results

The system generates:
1. **Fault Distribution Analysis**:
   - Distribution plots
   - Fault count statistics

2. **Performance Metrics**:
   - Individual fault F1 scores
   - Confusion matrices
   - Overall system performance

3. **Visualizations**:
   - Confusion matrix plots
   - Performance trend graphs

## Important Notes

1. **GPU Usage**: 
   - Default device is 'cuda:3'
   - Modify device parameter in code for different GPU or CPU usage

2. **Data Format**:
   - Input data should be in pickle format
   - Expected structure: {'data': X, 'label': y, 'README': info}

3. **Fault Types**:
   - Handles 4 fault types (including normal operation)
   - Codes: [0.0, 55.0, 67.0, 95.0]

## Contributing

Feel free to submit issues and enhancement requests!
