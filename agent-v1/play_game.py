#!/usr/bin/env python3

import sys
import time
import numpy as np
import argparse as ap
from enum import Enum, auto

import gym
import gym_mad
import agent as ag
from agent import RandomAgent
from dqn import DQNAgent
from observer import Observer
import torch

class AgentType(Enum):
    human = auto()
    random = auto()
    dqn = auto()

    def __str__(self):
        # Here self is the member
        return self.name

agent_choices = [name for name, member in AgentType.__members__.items()]

def parse_args():
    parser = ap.ArgumentParser(description="Script to test an agent in the mad-v1 gym env")
    parser.add_argument('--env_conf',type=str,default="default.json")

    parser.add_argument('--agent_a',type=str,choices=agent_choices,default=str(AgentType.random))
    parser.add_argument('--agent_b',type=str,choices=agent_choices,default=str(AgentType.random))
    parser.add_argument('--agent_a_path',type=str,default=None)
    parser.add_argument('--agent_b_path',type=str,default=None)
    parser.add_argument('--turn_delay','-d',type=float,default=0)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_episodes',type=int,default=1000)
    parser.add_argument('--eval_freq',type=int,default=4)

    parser.add_argument('-v',action='count',default=0,help="Verbose")

    parser.add_argument('--dqn_eps', type=float,default=0.1)
    parser.add_argument('--dqn_lr', type=float,default=0.0001)
    parser.add_argument('--dqn_discount', type=float, default=0.99)
    parser.add_argument('--dqn_buffer_size', type=int, default=50000)
    parser.add_argument('--dqn_buffer_batch', type=int, default=64)
    parser.add_argument('--dqn_buffer_burn_in', type=int, default=300)
    parser.add_argument('--no_dqn_burn_in_bar',action='store_false')
    parser.add_argument('--dqn_target_update_period',type=int,default=50)
    parser.add_argument('--dqn_model_hidden_size',type=int,default=32)
    parser.add_argument('--dqn_model_num_layers',type=int,default=3)
    parser.add_argument('--no_cuda', action="store_true")
    return parser.parse_args()

def get_player(env, agent_type_str:str, path:str, args) -> ag.MadAgent_v1:
    if agent_type_str == str(AgentType.human):
        raise ValueError("Human player not yet implemented")
    elif agent_type_str == str(AgentType.random):
        agent = RandomAgent(env.observation_size, env.action_size)
    elif agent_type_str == str(AgentType.dqn):
        agent = DQNAgent(
                        observation_size = env.observation_size, 
                        action_size = env.action_size,
                        epsilon = args.dqn_eps,
                        optimizer_lr = args.dqn_lr,
                        discount = args.dqn_discount,
                        buffer_size = args.dqn_buffer_size,
                        buffer_batch_size = args.dqn_buffer_batch,
                        buffer_burn_in = args.dqn_buffer_burn_in,
                        burn_in_bar = args.no_dqn_burn_in_bar,
                        target_update_period = args.dqn_target_update_period,
                        model_hidden_size = args.dqn_model_hidden_size,
                        model_num_layers = args.dqn_model_num_layers,
                        no_cuda = args.no_cuda)
    agent.initialize()
    return agent

def printif(*args, flag=True):
    if flag:
        print(*args, flush=True)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    env = gym.make("mad-v1")
    env.set_config_path(args.env_conf)
    observations = env.reset()
    done = False

    observer = Observer(C=env.config)

    # Currently setup for two AI playing against each other
    # Seperate AI for each player
    agent_a = get_player(env, args.agent_a, args.agent_a_path, args)
    agent_b = get_player(env, args.agent_b, args.agent_b_path, args)

    episode = 0
    is_burn_in_episode = False
    total_episodes = (args.train_episodes * (args.eval_freq+1)) // args.eval_freq
    sampled_episodes_for_plotting = [0, 500, 1000]

    while episode < total_episodes:

        is_burn_in_episode = False
        if agent_a.is_burning_in or agent_b.is_burning_in:
            episode = 0
            is_burn_in_episode = True

        is_eval_episode = False
        agent_a.train()
        agent_b.train()
        observer.plotting = False
        if not is_burn_in_episode and episode % args.eval_freq == 0:
            is_eval_episode = True
            agent_a.eval()
            agent_b.eval()
            if args.v>=2: observer.plotting = True
        
        PRINT = args.v>=3 and is_eval_episode and not is_burn_in_episode
        env.set_show_bar(show=not is_burn_in_episode and args.v >=1, e = episode)

        observations = env.reset()
        done = False

        while not done:

            printif("------------------------------------",flag=PRINT)
            printif(f"Begin {env.current_player}'s turn",flag=PRINT)
            if env.current_player == env.agent_a:
                action_vec = agent_a(observations[env.agent_a])
                new_observations, reward, done, info = env.step(action_vec)
                printif(f"Player took action {info['action'].action_str}",flag=PRINT)
                printif(info["turn_desc"],flag=PRINT)
                printif("New State:",flag=PRINT)
                env.render() if PRINT else None
                agent_a.report_SARS(
                    observations[env.agent_a],
                    action_vec, reward, new_observations[env.agent_a], done)
            else:
                action_vec = agent_b(observations[env.agent_b])
                new_observations, reward, done, info = env.step(action_vec)
                printif(f"Player took action {info['action'].action_str}",flag=PRINT)
                printif(info["turn_desc"],flag=PRINT)
                printif("New State:",flag=PRINT)
                env.render() if PRINT else None
                agent_b.report_SARS(
                    observations[env.agent_b],
                    action_vec, reward, new_observations[env.agent_b], done)

            time.sleep(args.turn_delay)
            observer.report_turn(env.S, action_vec, reward, info, done)
            observations = new_observations

            printif(f"End turn {info['turn_count']}\n",flag=PRINT)

        # end while not done
        episode += 1
        # if args.v>=1 and not is_burn_in_episode:
        #     printif("Episode completed: [" + str(episode) + "/" + str(total_episodes) + "]", flag=True)
        agent_a.report_new_episode()
        agent_b.report_new_episode()

        printif(f"Game Over! {info['winner']} won!",flag=PRINT)

    if args.v >= 2:
        print("Final State:")
        env.render()


    observer.print_final_stats()

if __name__ == "__main__":
    main()
