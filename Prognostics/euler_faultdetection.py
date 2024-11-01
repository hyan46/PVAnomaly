# -*- coding: utf-8 -*-
"""
Euler Characteristic-based Fault Detection with Bayesian Neural Networks
"""

import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import umap
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

# TensorFlow Probability aliases
tfd = tfp.distributions

class EulerFaultDetector:
    def __init__(self, num_points=20, filtration_start=0.0, filtration_end=30):
        self.num_points = num_points
        self.filtration_start = filtration_start
        self.filtration_end = filtration_end
        self.filtrations = np.linspace(filtration_start, filtration_end, num_points)
        
    def compute_euler_characteristic(self, data):
        """
        Compute the Euler characteristic for a given dataset.
        
        Args:
            data (np.ndarray): Input time series data
            
        Returns:
            np.ndarray: Euler characteristic values
        """
        cubeplex = gd.CubicalComplex(
            dimensions=[np.shape(data)[0], 1], 
            top_dimensional_cells=np.ndarray.flatten(data)
        )
        cubeplex.persistence()
        
        b = np.zeros((self.num_points, 3))
        ec = np.zeros(self.num_points)
        
        for i, fval in enumerate(np.flip(self.filtrations)):
            betti = cubeplex.persistent_betti_numbers(fval, fval)
            b[i] = [betti[0], betti[1], betti[2]]
            ec[i] = betti[0] - betti[1] + betti[2]
            
        return ec
    
    def process_dataset(self, X, segment_size=96, num_segments=10):
        """
        Process multiple time series and compute their Euler characteristics.
        
        Args:
            X (np.ndarray): Input dataset of shape (n_samples, sequence_length)
            segment_size (int): Size of each segment
            num_segments (int): Number of segments to process
            
        Returns:
            np.ndarray: Processed Euler characteristics
        """
        n_samples = X.shape[0]
        X_ec = np.zeros([n_samples, self.num_points * num_segments])
        
        for i in range(n_samples):
            for j in range(num_segments):
                start_idx = segment_size * j
                end_idx = segment_size * (j + 1)
                ec = self.compute_euler_characteristic(X[i, start_idx:end_idx])
                X_ec[i, self.num_points*j:self.num_points*(j+1)] = ec
                
        return X_ec

class BayesianFaultPredictor:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self._build_model()
    
    def _build_model(self):
        # Define the prior and posterior distributions
        def prior(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            return tf.keras.Sequential([
                tf.keras.layers.Dense(
                    n,
                    kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
                    bias_initializer=tf.keras.initializers.RandomNormal(0., 0.1)
                )
            ])

        def posterior(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            return tf.keras.Sequential([
                tf.keras.layers.Dense(
                    n,
                    kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
                    bias_initializer=tf.keras.initializers.RandomNormal(0., 0.1)
                )
            ])

        # Build the model using Keras functional API
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # First dense layer with uncertainty
        x = tf.keras.layers.Dense(
            64, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(inputs)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        # Second dense layer with uncertainty
        x = tf.keras.layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(1)(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def train(self, X_train, y_train, batch_size=64, epochs=1000, validation_split=0.2):
        # Compile model with custom loss to account for uncertainty
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        # Train the model
        return self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=50,
                    restore_best_weights=True
                )
            ]
        )
    
    def predict(self, X, num_samples=100):
        """
        Make predictions with uncertainty estimation using Monte Carlo Dropout
        """
        predictions = []
        for _ in range(num_samples):
            pred = self.model(X, training=True)  # Enable dropout during prediction
            predictions.append(pred)
        
        # Calculate mean and standard deviation of predictions
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0).flatten()  # Flatten to 1D array
        std_pred = np.std(predictions, axis=0).flatten()    # Flatten to 1D array
        
        return mean_pred, std_pred