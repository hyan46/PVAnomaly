# Maintenance Optimization using Reinforcement Learning

This project implements a maintenance optimization system using Deep Q-Learning to determine optimal maintenance policies for equipment based on Weibull failure analysis.

## Project Structure

```
.
├── data_preprocessing.py      # Processes maintenance logs and trains Weibull parameters
├── maintenance_optimization.py # Implements RL agent for maintenance optimization
├── processed_maintenance_log.csv # Input maintenance data (required)
├── model_params/             # Directory for saved models and parameters
│   ├── weibull_parameters.json  # Trained Weibull parameters
│   ├── maintenance_policy.pth   # Trained RL policy network
│   ├── policy_map.npy          # Numpy array of policy decisions
│   └── training_stats.json     # Training statistics
└── plots/                    # Directory for generated plots
    ├── training_history.png    # Training reward history
    └── policy_map.png          # Visualization of maintenance policy
```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- PyMC
- Gym
- Scipy

Install requirements using:
```bash
pip install torch numpy pandas matplotlib pymc gym scipy
```

## Usage

1. Data Preprocessing:
   ```bash
   python data_preprocessing.py
   ```
   This script:
   - Loads maintenance log data
   - Calculates equipment lifespans
   - Trains Weibull parameters
   - Saves parameters to `weibull_parameters.json`

2. Training the RL Agent:
   ```bash
   python maintenance_optimization.py
   ```
   This script:
   - Loads trained Weibull parameters
   - Trains a DQN agent for maintenance optimization
   - Saves the best model based on average reward
   - Generates policy maps and training visualizations

## Input Data Format

The `processed_maintenance_log.csv` should contain the following columns:
- REPORTDATE: Date of maintenance event
- Inverter: Equipment identifier
- Module: Component identifier
- PROBLEMCODE: Type of maintenance (should include 'FANS')

## Output Files

### Model Parameters
- `weibull_parameters.json`: Contains trained Weibull distribution parameters
- `maintenance_policy.pth`: Saved PyTorch model of the best performing policy
- `training_stats.json`: Training statistics including best average reward

### Visualizations
- `training_history.png`: Plot of training rewards and moving average
- `policy_map.png`: Visualization of the maintenance policy decisions

## Model Details

### Environment
- State space: (stage, shape, scale)
- Action space: 
  - 0: Preventive maintenance
  - 1: Corrective maintenance
  - 2: Continue production

### Rewards
- Production gain: +5
- Preventive maintenance cost: -10
- Corrective maintenance cost: -40
- Additional penalties based on reliability

### Training Parameters
- Episodes: 2000
- Batch size: 128
- Gamma (discount factor): 0.999
- Epsilon decay: 200
- Target network update: Every 10 episodes


## Notes

- Training results may vary due to the stochastic nature of RL
- The policy map shows the recommended actions at different stages and times
- Reliability threshold for forced maintenance is set at 0.8

## Logging

The training process is logged to:
- Console output
- `training.log` file

Check these logs for training progress and any potential errors.

