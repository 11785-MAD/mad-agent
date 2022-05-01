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
from tqdm import tqdm

class DeepModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size,num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_size = hidden_size
            out_size = hidden_size
            if i == 0: in_size = input_size
            if i == num_layers -1: out_size = output_size

            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_size
        x = self.layers(x)
        return x

class QNetwork(nn.Module):
    '''
    Class to be the network that the model uses for Q(s,a)
    '''
    def __init__(self, input_size, output_size, lr, hidden_size, num_layers, no_cuda):
        super().__init__()
        cuda = torch.cuda.is_available() and not no_cuda
        self.device = torch.device("cuda" if cuda else "cpu")
        self.model = DeepModel(input_size, output_size, hidden_size, num_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)

    def forward(self, state:np.ndarray) -> torch.Tensor:
        x = torch.Tensor(state).to(self.device)  # TODO: send to correct device
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
            np.array(states).astype(float),
            np.array(actions).astype(float),
            np.array(rewards).astype(float),
            np.array(next_states).astype(float),
            np.array(is_terminals).astype(float)
        )

class ReplayBuffer():
    '''
    Class which stores a history of transitions
    '''
    def __init__(self, buffer_size:int, batch_size:int) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.queue = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.queue)

    def append(self, t:Transition) -> None:
        self.queue.append(t)

    def sample_batch(self, batch_size = None) -> TransitionList:
        if batch_size is None:
            batch_size = self.batch_size

        batch_size = min(batch_size, len(self.queue))

        # TODO: use random.choice instead
        indices = np.random.randint(low=0, high=len(self.queue), size=batch_size)
        transitions = TransitionList()

        for i in indices:
            t = self.queue[i]
            transitions.append(t)

        return transitions

class DQNAgent(agent.MadAgent_v0):
    def __init__(self,
                 observation_size:int, 
                 action_size:int,
                 epsilon:float = 0.05,
                 optimizer_lr:float = 5e-4,
                 discount:float = 0.99,
                 buffer_size:int = 50000,
                 buffer_batch_size:int = 32,
                 buffer_burn_in:int = 300,
                 burn_in_bar = True,
                 target_update_period = 50,
                 model_hidden_size = 34,
                 model_num_layers = 3,
                 no_cuda = False,
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
        self.target_update_period = target_update_period

        self.Q_w = QNetwork(observation_size, action_size, optimizer_lr, model_hidden_size, model_num_layers, no_cuda)
        self.Q_target = QNetwork(observation_size, action_size, optimizer_lr,model_hidden_size, model_num_layers, no_cuda)
        self.R = ReplayBuffer(buffer_size, buffer_batch_size)

        self.c = 0 # Step counter
        self.episodes_seen = 0
        self.buffer_burn_in = buffer_burn_in
        self.is_burning_in = buffer_burn_in > 0

        if burn_in_bar:
            self.burn_in_bar = tqdm(
                total=self.buffer_burn_in, 
                dynamic_ncols=False, 
                leave=True,
                # position=0, 
                desc=f'DQN Burn in'
            )
        else:
            self.burn_in_bar = None

    def set_Q_target(self):
        self.Q_target.model = copy.deepcopy(self.Q_w.model)
        self.Q_target.eval()
        # TODO: set requires grad to be false
        
    def policy_random(self):
        action_idx = np.random.randint(0, self.action_size)
        return one_hot(self.action_size, action_idx)

    def policy_epsilon_greedy(self, q_vals:np.ndarray) -> int:
        if np.random.uniform() < self.epsilon:
            return self.policy_random()
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
        not_is_terminal_vec = torch.Tensor(np.logical_not(is_terminals)).to(self.Q_w.device) # [batch]
        target = torch.from_numpy(rewards).to(self.Q_w.device) + torch.mul(self.discount*max_Q_target, not_is_terminal_vec)  # [batch]

        # Calculate predicted
        q = self.Q_w(states)  # [batch x action_size]
        (predicted, predicted_indices) = torch.max(q*torch.from_numpy(actions).to(self.Q_w.device), dim=1)  # [batch]

        # Calculate loss
        loss = torch.nn.functional.mse_loss(target,predicted)  # []

        return loss
    
    def initialize(self):
        self.c = 0
        self.episodes_seen = 0

    def report_new_episode(self):
        self.episodes_seen += 1
        if self.burn_in_bar is not None:
            self.burn_in_bar.set_postfix(buffer=f"{len(self.R)}/{self.R.buffer_size}")
            self.burn_in_bar.update()
        if self.episodes_seen > self.buffer_burn_in and self.is_burning_in:
            self.is_burning_in = False
            self.c = 0
            if self.burn_in_bar is not None:
                self.burn_in_bar.close()
            print(f"DQN finished {self.buffer_burn_in} burn in episodes")

    def choose_action(self, observation):
        assert len(observation) == self.observation_size
        if self.is_burning_in:
            return self.policy_random()

        q_vals = self.Q_w(observation).detach().cpu().numpy()
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

        self.c += 1