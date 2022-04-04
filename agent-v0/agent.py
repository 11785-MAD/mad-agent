import numpy as np

class MadAgent_v0:
    def __init__(self, observation_size, action_size):
        self.observation_size = observation_size
        self.action_size = action_size

    def choose_action(self, observation):
        pass

    def report_SARS(self, observation, action, reward, new_observation):
        pass

    def __call__(self, x):
        return self.choose_action(x)

class RandomAgent(MadAgent_v0):
    def __init__(self, observation_size, action_size):
        super().__init__(observation_size, action_size)

    def choose_action(self, observation):
        # Choose a random action
        action_vec = np.zeros((self.action_size))
        action_idx = np.random.randint(0, self.action_size)
        action_vec[action_idx] = 1
        return action_vec