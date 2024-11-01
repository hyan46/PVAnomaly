import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import weibull_min
from collections import deque
import random
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from maintenance_optimization import MaintenanceOptimizer

# Constants
PREVENTIVE_COST = 10
CORRECTIVE_COST = 40
PRODUCTION_GAIN = 5
N_DAYS = 24

class MaintenanceEnv(gym.Env):
    def __init__(self, weibull_params=None):
        super(MaintenanceEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0: Preventive, 1: Corrective, 2: Produce
        self.observation_space = spaces.Box(low=np.array([0,0]), high=np.array([N_DAYS, 10]), dtype=np.float32)
        self.weibull_params = weibull_params  # Can be updated with fitted parameters
        self.reset()

    def step(self, action):
        stage, shape, scale = self.state
        reliability = 1 - weibull_min.cdf(stage, shape, scale=scale - 0.5*self.current_day)
        cost = 0

        if reliability < 0.8:
            action = 1

        if action == 0:  # Preventive maintenance
            preventive_impact = np.random.choice([0,1,2,3])
            stage = max(0, stage - preventive_impact)
            cost = -PREVENTIVE_COST-100*(1-reliability)**2
        elif action == 1:  # Corrective maintenance
            stage = 0
            cost = -CORRECTIVE_COST-800*(1-reliability)**2
        elif action == 2:  # Production
            stage += 1
            cost = PRODUCTION_GAIN
        
        self.state = (stage, shape, scale)
        done = self.current_day >= N_DAYS - 1
        self.current_day += 1

        return np.array(self.state), cost, done, {}

    def reset(self):
        self.current_day = 0
        if self.weibull_params is not None:
            shape, scale = self.weibull_params[self.current_day]
        else:
            shape, scale = 0.76, 3.64 * 24
        self.state = (0, shape, scale)
        return np.array(self.state)

# [Previous PolicyNet, ReplayBuffer, and MaintenanceOptimizer classes remain the same]
# ... [Rest of the code from maintenance_optimization.py]

if __name__ == "__main__":
    from data_processor import MaintenanceDataProcessor
    
    # Process data first
    processor = MaintenanceDataProcessor()
    _, _, weibull_params = processor.load_and_process_data('processed_maintenance_log.csv')
    
    # Configuration
    config = {
        'episodes': 2000,
        'batch_size': 128,
        'gamma': 0.999,
        'epsilon_start': 0.9,
        'epsilon_end': 0.05,
        'epsilon_decay': 200,
        'target_update': 10,
        'buffer_size': 10000
    }

    # Initialize and train the optimizer
    optimizer = MaintenanceOptimizer(config, weibull_params=weibull_params)
    optimizer.train() 