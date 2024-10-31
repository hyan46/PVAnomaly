# Tensor-Based Anomaly Detection

This project implements a tensor-based anomaly detection method using Hotelling's T² statistic and Tucker decomposition. The implementation combines dimensionality reduction through tensor decomposition with statistical process control methods to identify anomalies in tensor-structured data.

## Overview

The `TensorHotellingT2` class provides a robust framework for detecting anomalies in tensor data using two different methods:
- Hotelling's T² statistic
- Residual Sum of Squares (RSS)

## Features

- Non-negative Tucker decomposition for tensor dimensionality reduction
- Multiple detection methods:
  - T² statistic-based detection
  - Residual-based detection
  - Combined approach using both methods
- Empirical threshold calculation using quantile-based Upper Control Limits (UCL)
- Compatible with scikit-learn's estimator interface

## Requirements

- numpy
- tensorly
- matplotlib
- scikit-learn

## Installation 
```bash
pip install numpy tensorly matplotlib scikit-learn
```
## Usage

Here's a basic example of how to use the anomaly detector:
```python
from tensordetect import TensorHotellingT2
# Initialize the model
model = TensorHotellingT2(rank=[3, 3, 3], alpha=0.05)
# Fit the model on normal data
model.fit(X_train)
# Detect anomalies using T² method
predictions = model.predict(X_test, method='T2')
# Alternatively, use residual method
predictions = model.predict(X_test, method='residual')
# Or use both methods combined
predictions = model.predict(X_test, method='both')
```

## Parameters

- `rank`: List or integer specifying the Tucker rank(s) for decomposition
- `alpha`: Significance level for the empirical threshold (default: 0.05)
- `n_iter_max`: Maximum iterations for decomposition (default: 100)
- `tol`: Convergence tolerance for decomposition (default: 1e-6)

## Methods

### `fit(X)`
Fits the model to the training tensor data X.

### `predict(X, method='T2')`
Predicts anomalies in new data. Available methods:
- 'T2': Uses Hotelling's T² statistic
- 'residual': Uses reconstruction error
- 'both': Combines both methods

### `score_samples(X)`
Computes the T² statistic for each tensor in X.

### `reconstruction_error(X)`
Computes the residual sum of squares for each tensor in X.

## Output

The model returns:
- 1 for normal samples
- -1 for anomalies

## Performance Metrics

The implementation includes F1-score calculation to evaluate detection performance. The example code demonstrates the calculation of F1-scores for all three detection methods (T², residual, and combined approach).

## License

This project is open-source and available under the MIT License.