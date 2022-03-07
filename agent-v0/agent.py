import numpy as np


class MadAgent_v0:
    def __init__(self, observation_size, action_size):
        self.observation_size = observation_size
        self.action_size = action_size

    def choose_action(self, observation):
        return self.choose_random_action()

    def choose_random_action(self):
        action_vec = np.zeros((5))
        action_idx = np.random.randint(0, 5)
        action_vec[action_idx] = 1
        return action_vec

    def report_SARS(self, observation, action, reward, new_observation):
        # TODO
        pass

    def __call__(self, x):
        return self.choose_action(x)
