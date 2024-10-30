# Skorch-PyTorch Integration for PRBNN

## Introduction
This code integrates Skorch to wrap PyTorch models, enhancing compatibility with scikit-learn-like operations. This approach simplifies the machine learning workflow, making it more accessible and efficient.

## Installation
Ensure you have Anaconda installed, and then follow these steps to set up the necessary environment:

```bash
# Create the conda environment from the environment.yml file
conda env create -f environment.yml
```

# Activate the conda environment
conda activate your-env-name
Note: Replace your-env-name with the name of your environment as specified in the environment.yml file.


## Setting Up the Model
Model configuration requires minimal adjustments, focusing on essential parameters such as:

Learning Rate: Adjust this to optimize training and convergence.
Computation Device: Use cuda for GPU and cpu for CPU.
save_path: Specify the directory for saving model checkpoints.
data_path: Specify the directory for accessing the dataset.

## Usage
### Training
To train the model, utilize the .fit() method, which mirrors the simplicity and usability of scikit-learn methods:
```
model.fit(X_train, y_train)
```
### Prediction
For making predictions with the trained model:
```
predictions = model.predict(X_test)
```
