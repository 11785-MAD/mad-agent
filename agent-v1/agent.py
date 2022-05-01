import numpy as np
import torch.nn as nn
        
def one_hot(action_size:int, action_index:int) -> np.ndarray:
    action_vec = np.zeros(action_size)
    action_vec[action_index] = 1
    return action_vec

class MadAgent_v1(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.is_burning_in = False
        self.last_loss = 0.0

    def forward(self, x):
        return self.choose_action(x)

    def initialize(self):
        pass

    def report_new_episode(self):
        pass

    def choose_action(self, observation):
        pass

    def report_SARS(self, observation, action, reward, new_observation, is_terminal):
        pass

class RandomAgent(MadAgent_v1):
    def __init__(self, observation_size, action_size):
        super().__init__(observation_size, action_size)

    def choose_action(self, observation):
        # Choose a random action
        action_idx = np.random.randint(0, self.action_size)
        return one_hot(self.action_size, action_idx)