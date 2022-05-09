import numpy as np
import torch.nn as nn
from gym_mad.envs.mad_env_v1 import MadEnv_v1, MadState_v1, MadAction_v1, MadGameConfig_v1
        
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
        self.last_loss = -1

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

class RandomValidAgent(MadAgent_v1):
    def __init__(self, observation_size, action_size, config):
        super().__init__(observation_size, action_size)
        self.config = config

    def choose_action(self, observation):
        valid_actions = []
        S = MadState_v1(self.config)
        for A_idx in range(MadAction_v1.action_size):
            A = MadAction_v1(one_hot(MadAction_v1.action_size, A_idx))
            S.data = observation.copy()
            S.cash_a *= S.config.data["max_cash"]
            S.cash_b *= S.config.data["max_cash"]
            S.income_a *= S.config.data["max_income"]
            S.income_b *= S.config.data["max_income"]
            S.military_a *= S.config.data["max_military"]
            S.military_b *= S.config.data["max_military"]
            reward, done, winner, info = A.apply_dynamics(S, self.config)
            if reward/self.config.data["reward_scale"] == self.config.data["invalid_penalty"] or reward/self.config.data["reward_scale"] == self.config.data["over_max_penalty"]:
                continue
            valid_actions.append(A_idx)

        if len(valid_actions) > 0:
            action_idx = np.random.choice(np.array(valid_actions))
        else:
            action_idx = np.random.randint(0, self.action_size)
        return one_hot(self.action_size, action_idx)
