'''
Author: Calen Robinson
This file is based on Calen's HW2 of 10703 Deep Reinforcement Learning Fall 2021 at CMU
'''
import gym

import agent
from agent import one_hot
import torch
import torch.nn as nn
import numpy as np
import copy
import collections

class DeepModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.temp = nn.Linear(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 1
        assert x.shape[0] == self.input_size

        x = self.temp(x)
        return x

class QNetwork(nn.Module):
    '''
    Class to be the network that the model uses for Q(s,a)
    '''
    def __init__(self, input_size, output_size, lr):
        super().__init__()
        self.model = DeepModel(input_size, output_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)

    def forward(self, state:np.ndarray) -> torch.Tensor:
        x = torch.Tensor(state)
        return self.model(x)

    def save_state_dict(self, path:str) -> None:
        pass

    def load_state_dict(self, path:str) -> None:
        pass

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
            self.action,
            self.reward,
            self.next_state,
            self.is_terminal,
        )


class TransitionList():
    def __init__(self, transitions:list = []):
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

class ReplayBuffer():
    '''
    Class which stores a history of transitions
    '''
    def __init__(self, buffer_size:int, batch_size:int) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.burn_in = burn_in
        self.queue = collections.deque(maxlen=buffer_size)

    def append(self, t:Transition) -> None:
        self.queue.append(t)

    def sample_batch(self, batch_size = None) -> TransitionList:
        if batch_size is None:
            batch_size = self.batch_size

        batch_size = min(batch_size, len(self.queue))

        indices = np.random.randint(low=0, high=len(self.queue))
        transitions = TransitionList()

        for i in indices:
            t = self.queue[i]
            transitions.append(t)

        return transitions

class DQNAgent(agent.MadAgent_v0):
    def __init__(self,
                 observation_size:int, 
                 action_size:int,
                 epsilon:float,
                 optimizer_lr:float,
                 discount:float,
                 buffer_size:int,
                 buffer_batch_size:int,
                 buffer_burn_in:int,
                 ):
        '''
        Args:
            observation_size:int - The size of the observations
            action_size:int - The size of the action space
            epsilon:float - value for epsilon greedy policy
        '''
        super().__init__(observation_size, action_size)
        self.epsilon = epsilon
        self.discount = discount

        self.Q_w = QNetwork(observation_size, action_size, optimizer_lr)
        self.Q_target = QNetwork(observation_size, action_size, optimizer_lr)
        self.R = ReplayBuffer(buffer_size, buffer_batch_size)

        self.c = 0 # Step counter
        self.episodes_seen = 0
        self.buffer_burn_in = buffer_burn_in
        self.is_burning_in = buffer_burn_in > 0

    def set_Q_target(self):
        self.Q_target.model = copy.deepcopy(self.Q_w.model)
        self.Q_target.eval()
        
    def policy_random(self):
        action_idx = np.random.randint(0, self.action_size)
        return one_hot(self.action_size, action_idx)

    def policy_epsilon_greedy(self, q_vals:np.ndarray) -> int:
        if np.random.uniform() < self.epsilon:
            return one_hot(self.action_size, np.random.randint(0, len(q_vals)))
        return self.policy_greedy(q_vals)

    def policy_greedy(self, q_vals:np.ndarray) -> np.ndarray:
        return one_hot(self.action_size, np.argmax(q_vals))

    def loss(self, transitions:TransitionList) -> torch.Tensor:
        (states, # [batch x observation_size]
         actions, # [batch x action_size]
         rewards, # [batch]
         next_states, # [batch x observation_size]
         is_terminals # [batch]
        ) = transitions.unpack() # All are np.ndarray

        # Calculate Target
        (max_Q_target, max_indices_Q_target) = torch.max( self.Q_target(next_states), dim=1) # [batch]
        not_is_terminal_vec = torch.Tensor(np.logical_not(is_terminals).float()) # [batch]
        target = rewards + torch.mul(self.discount*max_Q_target, not_is_terminal_vec)  # [batch]

        # Calculate predicted
        q = self.Q_w(states)  # [batch x action_size]
        (predicted, predicted_indices) = torch.max(q*actions, dim=1)  # [batch]
        loss = torch.nn.functional.mse_loss(target,predicted)  # []

        return loss
    
    def initialize(self, env):
        self.c = 0
        self.episodes_seen = 0

    def report_new_episode(self):
        self.episodes_seen += 1
        if self.episodes_seen > self.buffer_burn_in:
            self.is_burning_in = False

    def choose_action(self, observation):
        assert len(observation) == self.observation_size
        q_vals = self.Q_w(observation)
        if self.is_burning_in:
            return self.policy_random()

        if self.training:
            return self.policy_epsilon_greedy(q_vals)

        return self.policy_greedy(q_vals)

    def report_SARS(self, observation, action, reward, new_observation, is_terminal):
        if not self.training:
            return

        t = Transition(observation, action, reward, new_observation, is_terminal)
        self.R.append(t)

        if self.is_burning_in:
            return

        self.Q_w.train()
        transitions = self.R.sample_batch()
        self.Q_w.optimizer.zero_grad()
        L = self.loss(transitions)
        L.backward()
        self.Q_w.optimizer.step()
        self.Q_w.eval()

        if self.c % self.target_update_period == 0:
            self.set_Q_target()