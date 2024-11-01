# PV System Health Management Framework

A comprehensive framework for PV system health management, including anomaly detection, prognostics, power prediction, and maintenance optimization.

## Project Structure

```
.
├── Anomaly Detection/          # Tensor-based anomaly detection
├── Prognostics/               # Fault detection using Euler characteristics
├── Power Prediction/          # Power prediction using PRBNN
├── Maintenance_Optimization/  # RL-based maintenance scheduling
└── README.md
```

## Module Descriptions

### 1. Anomaly Detection
Implements tensor-based anomaly detection using Hotelling's T² statistic and Tucker decomposition. Features:
- Non-negative Tucker decomposition for dimensionality reduction
- Multiple detection methods (T² statistic, Residual Sum of Squares)
- Empirical threshold calculation

[Detailed Anomaly Detection Documentation](Anomaly%20Detection/README.md)

### 2. Prognostics
Implements fault detection using topological data analysis (Euler characteristics) with Bayesian Neural Networks. 

![Fault Detection Results](Prognostics/plots/fault_detection_results.png)

The visualization shows:
- Training convergence (left)
- Prediction accuracy with uncertainty (middle)
- Distribution separation between normal and fault conditions (right)

[Detailed Prognostics Documentation](Prognostics/README.md)

### 3. Power Prediction
Provides power prediction using Probabilistic Residual Bayesian Neural Networks (PRBNN):
- Skorch-PyTorch integration for scikit-learn compatibility
- Uncertainty quantification in predictions
- Enhanced training workflow

[Detailed Power Prediction Documentation](Power%20Prediction/PRBNN_power_prediction/README.md)

### 4. Maintenance Optimization
Implements reinforcement learning for maintenance scheduling:

![Maintenance Policy](Maintenance_Optimization/plots/policy_map.png)

Features:
- Deep Q-Learning for maintenance policy optimization
- Weibull-based failure analysis
- Visual policy mapping for decision support

[Detailed Maintenance Optimization Documentation](Maintenance_Optimization/README.md)

## Key Features

1. **Uncertainty Quantification**
   - Bayesian approaches in fault detection
   - Probabilistic power predictions
   - Risk-aware maintenance decisions

2. **Visualization Tools**
   - Training progress monitoring
   - Uncertainty visualization
   - Policy mapping

3. **Integration**
   - Compatible with scikit-learn
   - Modular design
   - Flexible deployment options

## Documentation Structure

Each module has its own detailed documentation covering:
- Installation requirements
- Usage examples
- API reference
- Configuration options
- Performance metrics

Please refer to the individual README files linked above for detailed information about each component.

## References

[Add relevant papers and citations]

## License

This project is licensed under the MIT License - see the LICENSE file for details.
