import pandas as pd
import numpy as np
import pymc as pm
import random
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import math
import matplotlib.patches as mpatches
plt.rcParams['axes.unicode_minus'] = False

log = pd.read_csv('processed_maintenance_log.csv')
log.head()

# Filtering rows that contain 'FANS' in the 'PROBLEMCODE' column
igbt_log = log[log['PROBLEMCODE'] == 'FANS']

# Sorting the filtered data based on 'Inverter' and 'Module'
igbt_log = igbt_log.sort_values(by=['Inverter', 'Module'])

#Reindexing
igbt_log = igbt_log.reset_index(drop=True)
igbt_log

maintenance_index = igbt_log[['Inverter', 'Module']].duplicated(keep=False)
maintenance_log = igbt_log[maintenance_index].reset_index(drop=True)
maintenance_log['REPORTDATE'] = pd.to_datetime(maintenance_log['REPORTDATE'])


maintenance_log_shift = maintenance_log.shift(1)


replace_index = (maintenance_log[['Inverter', 'Module']] == maintenance_log_shift[['Inverter', 'Module']])


rows_same_as_previous = replace_index.all(axis=1)


date_diff = maintenance_log['REPORTDATE'] - maintenance_log_shift['REPORTDATE']


maintenance_log.loc[rows_same_as_previous, 'LifeSpan'] = date_diff[rows_same_as_previous]


replace_log = maintenance_log[rows_same_as_previous]

replace_log = replace_log.reset_index(drop=True)

replace_log

failure_log = [[] for _ in range(24)]

for index, row in replace_log.iterrows():

    month_index = (row['REPORTDATE'].year - 2019) * 12 + row['REPORTDATE'].month - 5
    if 0 <= month_index < 24:

        failure_log[month_index].append(row['LifeSpan'].days/365.25)

failure_log

import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

shape_prior, scale_prior = 0.76, 3.64           #1.35, 1.61

# failure_log = np.array([0.1, 0.75, 1.1, 0.2])

failure_ob = np.array([])
for _ in failure_log:
  if _:
    failure_ob = np.append(failure_ob,np.array(_))

print(failure_ob)

# failure_ob = np.array([0.05, 0.25, 0.13, 0.68, 0.13, 0.12, 0.13, 0.5, 0.033, 0.364, 0.14, 0.167, 0.063, 0.167,0.167, 0.548])

with pm.Model() as model:

    shape = pm.Normal('shape', mu=shape_prior, sigma=0.001)
    scale = pm.Normal('scale', mu=scale_prior, sigma=0.4)

    weibull = pm.Weibull('weibull', alpha=shape, beta=scale, observed=failure_ob)

    trace = pm.sample(1000, tune=2000, return_inferencedata=False)

import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

shape, scale = 0.76, 3.64           #1.35, 1.61

parameter_sc = np.zeros((24,2))
day = 0
for failure in failure_log:
  if failure:
    with pm.Model() as model:

        shape = pm.Normal('shape', mu=shape, sigma=0.001)
        scale = pm.Normal('scale', mu=scale, sigma=0.53)

        weibull = pm.Weibull('weibull', alpha=shape, beta=scale, observed=np.array(failure))

        trace = pm.sample(1000, tune=2000, return_inferencedata=False)
        shape = np.mean(trace['shape'])
        scale = np.mean(trace['scale'])

  parameter_sc[day][0]=shape
  parameter_sc[day][1]=scale
  day += 1

parameter_sc

"""RL"""

import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import weibull_min
from collections import deque
import random
import matplotlib.pyplot as plt

# Constants
PREVENTIVE_COST = 10
CORRECTIVE_COST = 40
PRODUCTION_GAIN = 5
N_DAYS = 24

# Environment
class MaintenanceEnv(gym.Env):
    def __init__(self):
        super(MaintenanceEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0: Preventive, 1: Corrective, 2: Produce
        self.observation_space = spaces.Box(low=np.array([0,0]), high=np.array([N_DAYS, 10]), dtype=np.float32)
        self.reset()

    def step(self, action):
        stage, shape, scale = self.state
        reliability = 1 - weibull_min.cdf(stage, shape, scale=scale - 0.5*self.current_day)
        cost = 0

        if reliability < 0.8:  # Forced corrective maintenance
            action = 1

        if action == 0:  # Preventive maintenance
            preventive_impact = np.random.choice([0,1,2,3])  # Randomize preventive impact
            stage = max(0, stage - preventive_impact)
            cost = -PREVENTIVE_COST-100*(1-reliability)**2
        elif action == 1:  # Corrective maintenance
            stage = 0
            cost = -CORRECTIVE_COST-800*(1-reliability)**2
        elif action == 2:  # Produce
            stage += 1
            cost = PRODUCTION_GAIN
        self.state = (stage, shape, scale)

        # Check if we are at the end of the period
        done = self.current_day >= N_DAYS - 1
        self.current_day += 1

        # For simplicity, the reward is the negative of the cost
        reward = cost

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.current_day = 0

        self.state = (0, 0.76, 3.64 * 24)  # Initial state
        return np.array(self.state)

# Neural Network for the RL Agent
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Experience Replay Buffer
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

# Select Action
def select_action(state, policy_net, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            return policy_net(state).max(0)[1].view(1, 1).item()
    else:
        return np.random.choice([0, 1, 2])

# Update Policy
def optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return

    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    state_batch = torch.tensor(states, dtype=torch.float32)
    action_batch = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    reward_batch = torch.tensor(rewards, dtype=torch.float32)
    next_state_batch = torch.tensor(next_states, dtype=torch.float32)
    done_batch = torch.tensor(dones, dtype=torch.bool)

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all non-terminal next states.
    non_final_mask = ~done_batch
    non_final_next_states = next_state_batch[non_final_mask]
    next_state_values = torch.zeros(batch_size)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Main Function
def train_agent(episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, target_update, buffer_size):
    env = MaintenanceEnv()
    policy_net = PolicyNet()
    target_net = PolicyNet()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayBuffer(buffer_size)

    steps_done = 0
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
                print(f'Episode {episode}, Total Reward: {total_reward}')
                break

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print("Training complete")
    output_policy_map(policy_net, env)

def output_policy_map(policy_net, env, n_stages=16, n_days=N_DAYS):
    policy_map = np.zeros((n_stages, n_days))

    for stage in range(n_stages):
        for day in range(n_days):
            # Assuming the state is defined by (stage, shape, scale)
            # Adjust this to match your actual state definition
            state = np.array([stage, 0.76, 3.64 * 24-day*0.5], dtype=np.float32)
            action = select_action(state, policy_net, epsilon=0)
            policy_map[stage, day] = action

    # Plotting the policy map
    plt.imshow(policy_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Policy Map')
    plt.xlabel('Month')
    plt.ylabel('Stage')
    plt.xticks(range(n_days))
    plt.yticks(range(n_stages))
    plt.show()

# Hyperparameters
EPISODES = 2000
BATCH_SIZE = 128
GAMMA = 0.999
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 200
TARGET_UPDATE = 10
BUFFER_SIZE = 10000

train_agent(EPISODES, BATCH_SIZE, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY, TARGET_UPDATE, BUFFER_SIZE)
