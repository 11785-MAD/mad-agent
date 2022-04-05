'''
Author: Calen Robinson
This file is based on Calen's HW2 of 10703 Deep Reinforcement Learning Fall 2021 at CMU
'''
import gym

import agent
import torch
import torch.nn as nn
import numpy as np
import copy
import collections

class DeepModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__():
        self.temp = nn.Linear(input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x:torch.Tensor) -> torch.Tensor
        assert len(x.shape) == 1
        assert x.shape[0] = self.input_size

        x = self.temp(x)
        return x

class QNetwork(nn.Module):
    '''
    Class to be the network that the model uses for Q(s,a)
    '''
    def __init__(self, input_size, output_size):
        super().__init__():
        self.model = DeepModel(input_size, output_size)

    def forward(self, state:np.ndarray) -> np.ndarray:
        x = torch.Tensor(state)
        return model(x).detach().cpu().numpy()

    def save_state_dict(self, path:str) -> None:
        pass

    def load_state_dict(self, path:str) -> None:

class Transition():
    '''
    Class to handle a SARS transition
    '''
    def __init__(self, state, action, reward, next_state, is_terminal):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_terminal = is_terminal

    def unpack(self) -> tuple:
        return (
            self.state,
            self.action = action,
            self.reward = reward,
            self.next_state = next_state,
            self.is_terminal = is_terminal,
        )


class TransitionList():
    def __init__(self, transitions:list[Transition] = []):
        self.transitions = transitions

    def append(self, T:Transition) -> None:
        self.transitions.append(T)

    def unpack(self) -> tuple:
        states = []
        actions = []
        rewards = []
        next_states = []
        is_terminals = []

        for t in self.transitions:
            (S,A,R,NS,IT) = t.unpack()
            states.append(S)
            actions.append(A)
            rewards.append(R)
            next_states.append(NS)
            is_terminals.append(IT)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(is_terminals)
        )

class MemoryBank():
    '''
    Class which stores a history of transitions
    '''
    def __init__(self, bank_size:int, burn_in:int, batch_size:int) -> None:
        self.bank_size = bank_size
        self.burn_in = burn_in
        self.batch_size = batch_size
        self.queue = collections.deque(maxlen=bank_size)

    def sample_batch(self, batch_size = None) -> TransitionList:
        if batch_size is None:
            batch_size = self.batch_size

        batch_size = min(batch_size, len(self.bank))

        indices = np.random.randint(low=0, high=len(self.queue))
        transitions = TransitionList()

        for i in indices:
            t = self.queue[i]
            transitions.append()

        return transitions

class DQNAgent(agent.madAgent_v0):
    def __init__(self,
                 observation_size:int, 
                 action_size:int,
                 epsilon:float):
        '''
        Args:
            observation_size:int - The size of the observations
            action_size:int - The size of the action space
            epsilon:float - value for epsilon greedy policy
        '''
        super().__init__(observation_size, action_size)
        self.epsilon = epsilon

        self.Q_w = Network(observation_size, action_size)
        self.Q_target Network(observation_size, action_size)

    def set_Q_target(self):
        self.Q_target.model = copy.deepcopy(self.Q_w.model)
        self.Q_target.eval()

    def policy_epsilon_greedy(self, q_vals:np.ndarray) -> int:
        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, len(q_vals))
        return self.policy_greedy()

    def policy_greedy(self, q_vals:np.ndarray) -> int:
        return np.argmax(q_val)

    def loss(self, transitions:TransitionList) -> torch.Tensor:
        (states, # [batch x observation_size]
         actions, # [batch]
         rewards, # [batch]
         next_states, # [batch x observation_size]
         is_terminals # [batch]
        ) = transitions.unpack() # All are np.ndarray



    def choose_action(self, observation):