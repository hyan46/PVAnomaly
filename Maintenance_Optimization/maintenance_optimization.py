import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import weibull_min
from collections import deque
import random
import json
import os
import sys
import logging
from datetime import datetime
# Add this right after the imports and before the Constants section
print("Starting maintenance optimization script...")


# Constants
PREVENTIVE_COST = 10
CORRECTIVE_COST = 40
PRODUCTION_GAIN = 5
N_DAYS = 24

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class MaintenanceEnv(gym.Env):
    def __init__(self):
        super(MaintenanceEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0: Preventive, 1: Corrective, 2: Produce
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([N_DAYS, 10, 10]), 
            dtype=np.float32
        )
        
        # Create model_params directory if it doesn't exist
        os.makedirs('model_params', exist_ok=True)
        
        # Try to load parameters from either location
        try:
            if os.path.exists('model_params/weibull_parameters.json'):
                param_path = 'model_params/weibull_parameters.json'
            else:
                raise FileNotFoundError("No weibull_parameters.json found!")
                
            logging.info(f"Loading parameters from {param_path}")
            with open(param_path, 'r') as f:
                params = json.load(f)
                
            self.global_shape = params['global_shape']
            self.global_scale = params['global_scale']
            self.monthly_parameters = np.array(params['monthly_parameters'])
            
        except Exception as e:
            logging.error(f"Error loading parameters: {str(e)}")
            raise
            
        self.reset()

    def step(self, action):
        stage, shape, scale = self.state
        reliability = 1 - weibull_min.cdf(stage, shape, scale=scale - 0.5*self.current_day)
        cost = 0

        if reliability < 0.8:  # Forced corrective maintenance
            action = 1

        if action == 0:  # Preventive maintenance
            preventive_impact = np.random.choice([0,1,2,3])
            stage = max(0, stage - preventive_impact)
            cost = -PREVENTIVE_COST-100*(1-reliability)**2
        elif action == 1:  # Corrective maintenance
            stage = 0
            cost = -CORRECTIVE_COST-800*(1-reliability)**2
        elif action == 2:  # Produce
            stage += 1
            cost = PRODUCTION_GAIN
        
        self.state = (stage, shape, scale)
        done = self.current_day >= N_DAYS - 1
        self.current_day += 1
        
        return np.array(self.state), cost, done, {}

    def reset(self):
        self.current_day = 0
        self.state = (0, self.global_shape, self.global_scale * 24)
        return np.array(self.state)

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

def select_action(state, policy_net, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            return policy_net(state).max(0)[1].view(1, 1).item()
    else:
        return random.randrange(3)

def optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return

    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    state_batch = torch.tensor(states, dtype=torch.float32)
    action_batch = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    reward_batch = torch.tensor(rewards, dtype=torch.float32)
    next_state_batch = torch.tensor(next_states, dtype=torch.float32)
    done_batch = torch.tensor(dones, dtype=torch.bool)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    non_final_mask = ~done_batch
    next_state_values = torch.zeros(batch_size)
    next_state_values[non_final_mask] = target_net(next_state_batch[non_final_mask]).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_agent(episodes=2000, batch_size=128, gamma=0.999, 
                epsilon_start=0.9, epsilon_end=0.05, 
                epsilon_decay=200, target_update=10, buffer_size=10000):
    
    try:
        logging.info("Starting training process...")
        
        # Create directories
        os.makedirs('model_params', exist_ok=True)
        
        # Save training parameters
        training_params = {
            'episodes': episodes,
            'batch_size': batch_size,
            'gamma': gamma,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay,
            'target_update': target_update,
            'buffer_size': buffer_size,
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save model architecture parameters
        model_architecture = {
            'input_size': 3,
            'hidden_size': 128,
            'output_size': 3,
            'optimizer': 'Adam'
        }
        
        # Combine all parameters
        all_params = {
            'training_params': training_params,
            'model_architecture': model_architecture
        }
        
        # Save parameters to JSON file
        params_save_path = 'model_params/model_parameters.json'
        with open(params_save_path, 'w') as f:
            json.dump(all_params, f, indent=4)
        logging.info(f"Saved model parameters to {params_save_path}")
        
        env = MaintenanceEnv()
        policy_net = PolicyNet()
        target_net = PolicyNet()
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.Adam(policy_net.parameters())
        memory = ReplayBuffer(buffer_size)

        rewards_history = []
        steps_done = 0
        
        logging.info(f"Starting training for {episodes} episodes...")
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            while True:
                epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                         np.exp(-1. * steps_done / epsilon_decay)
                action = select_action(state, policy_net, epsilon)
                next_state, reward, done, _ = env.step(action)
                
                memory.push(state, action, reward, next_state, done)
                state = next_state
                optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma)
                
                total_reward += reward
                steps_done += 1
                
                if done:
                    rewards_history.append(total_reward)
                    if episode % 100 == 0:
                        logging.info(f'Episode {episode}, Total Reward: {total_reward:.2f}')
                    break

            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Save the trained model and rewards history
        model_save_path = 'model_params/maintenance_policy.pth'
        torch.save({
            'model_state_dict': policy_net.state_dict(),
            'rewards_history': rewards_history
        }, model_save_path)
        logging.info(f"Saved model and rewards history to {model_save_path}")
        
        return True

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # First, make sure we have the parameters
        if not (os.path.exists('weibull_parameters.json') or 
                os.path.exists('model_params/weibull_parameters.json')):
            logging.error("No weibull parameters found! Please run data_preprocessing.py first.")
            sys.exit(1)
            
        logging.info("Starting maintenance optimization process...")
        train_agent()
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
        sys.exit(1)