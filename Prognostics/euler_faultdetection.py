# -*- coding: utf-8 -*-

!pip install gudhi persim mnist
!pip install umap-learn

# Import necessary libraries
import numpy as np
import gudhi as gd  # Assuming gudhi is imported as gd
from matplotlib import pyplot as plt
import umap

# Define the number of points, start, and end values for the filtration
numPoints = 20
filtrationStart = 0.0
filtrationEnd = 30

# Generate a set of filtration values evenly spaced between filtrationStart and filtrationEnd
filtrations = np.linspace(filtrationStart, filtrationEnd, numPoints)

# Set parameters s and u
s = 200
u = 45000

# Define a function to compute the Euler characteristic
def getEC(data):
    """
    Compute the Euler characteristic for a given dataset using persistent Betti numbers.

    Parameters:
    - data: Input dataset

    Returns:
    - ec: Array containing the Euler characteristic computed at different filtration levels
    """
    # Create a cubical complex from the given data
    cubeplex = gd.CubicalComplex(dimensions=[np.shape(data)[0], 1], top_dimensional_cells=np.ndarray.flatten(data))

    # Compute the persistence of the cubical complex
    cubeplex.persistence()

    # Initialize arrays to store persistent Betti numbers and Euler characteristic
    b = np.zeros((numPoints, 3))
    ec = np.zeros(numPoints)

    # Iterate over the filtrations in reverse order
    for (i, fval) in enumerate(np.flip(filtrations)):
        # Compute persistent Betti numbers for the current filtration value
        betti = cubeplex.persistent_betti_numbers(fval, fval)

        # Store the Betti numbers
        b[i] = [betti[0], betti[1], betti[2]]

        # Compute the Euler characteristic
        ec[i] = betti[0] - betti[1] + betti[2]

    # Return the Euler characteristic array
    return ec

# Initialize an array to store the computed Euler characteristics for all datasets
X_ec = np.zeros([231, 200])

# Loop over each dataset
for i in range(231):
    # Loop over each segment of the dataset
    for j in range(10):
        # Compute the Euler characteristic for the current segment and store it in X_ec
        X_ec[i, 20*j:20*(j+1)] = getEC(X[i, 96*j:96*(j+1)])

# Use UMAP for dimensionality reduction
X_embedding = umap.UMAP().fit_transform(X_ec)

# Plot the reduced-dimensional data points
plt.figure()
plt.scatter(X_embedding[:, 0], X_embedding[:, 1], s=1, cmap='Spectral')
plt.xticks([])  # Hide x-axis ticks
plt.yticks([])  # Hide y-axis ticks

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# TensorFlow Probability layers and distributions
tfpl = tfp.layers
tfd = tfp.distributions

# Placeholder for data paths
file_path_train = 'placeholder_train_data_path.txt'
file_path_test = 'placeholder_test_data_path.txt'
true_true_fault_file_path = 'placeholder_true_fault_path.txt'

# Load the training dataset
df_train = pd.read_csv(file_path_train, sep=' ', header=None, names=column_names, usecols=range(26))

# Calculate RUL for training data
df_train['Fault'] = df_train['PLACEHOLDER']

# Preprocessing for training data
features = [col for col in df_train.columns if col not in ['PLACEHOLDER']]
X_train = df_train[features]
y_train = df_train['PLACEHOLDER']

# Splitting dataset and normalizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

df_test = pd.read_csv(file_path_test, sep=' ', header=None, names=column_names, usecols=range(26))
X_test = df_test[features]
X_test_scaled = scaler.transform(X_test)

true_fault = pd.read_csv(true_true_fault_file_path, header=None, names=['true_fault'])

# Define Bayesian Neural Network model
def build_bnn_model(input_shape):
    """
    Build a Bayesian Neural Network model with variational layers.

    Parameters:
    - input_shape: Shape of the input data

    Returns:
    - model: TensorFlow model
    """
    def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
        """
        Define posterior distribution for weights using mean-field variational inference.
        """
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        return tf.keras.Sequential([
            tfpl.VariableLayer(2 * n, dtype=dtype),
            tfpl.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n],
                           scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                reinterpreted_batch_ndims=1)),
        ])

    def prior_trainable(kernel_size, bias_size=0, dtype=None):
        """
        Define trainable prior distribution for weights.
        """
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfpl.VariableLayer(n, dtype=dtype),
            tfpl.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t, scale=1),
                reinterpreted_batch_ndims=1)),
        ])

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tfpl.DenseVariational(units=64, make_posterior_fn=posterior_mean_field,
                              make_prior_fn=prior_trainable, kl_weight=1/X_train.shape[0], activation='relu'),
        tfpl.DenseVariational(units=1, make_posterior_fn=posterior_mean_field,
                              make_prior_fn=prior_trainable, kl_weight=1/X_train.shape[0]),
        tfpl.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
    ])

    return model

# Model compilation and training
model = build_bnn_model(input_shape=(X_train_scaled.shape[1],))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_scaled, y_train, batch_size=64, epochs=1000, verbose=1, validation_split=0.2)

# Predictions for the test set
y_pred_dist = model.predict(X_test_scaled)
y_pred = np.squeeze([dist.mean() for dist in y_pred_dist])  # Convert tensor predictions to numpy array

last_cycle_idx = df_test.groupby('Module')['time'].idxmax()
y_pred_last_cycle = y_pred[last_cycle_idx]

rmse = sqrt(mean_squared_error(true_rul['true_fault'], y_pred_last_cycle))
print(f'RMSE on test set: {rmse}')

# Plotting True RUL vs Predicted RUL
plt.figure(figsize=(10, 6))
plt.plot(true_rul['true_fault'].values, label='True Fault', marker='o')
plt.plot(y_pred_last_cycle, label='Predicted Fault', marker='x', linestyle='None')
plt.legend()
plt.show()