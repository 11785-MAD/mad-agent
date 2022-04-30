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

class AgentType(Enum):
    human = auto()
    random = auto()
    dqn = auto()

    def __str__(self):
        # Here self is the member
        return self.name

agent_choices = [name for name, member in AgentType.__members__.items()]

def parse_args():
    parser = ap.ArgumentParser(description="Script to test an agent in the mad-v0 gym env")
    parser.add_argument('--env_conf',type=str,default="default.json")
    parser.add_argument('--agent_a',type=str,choices=agent_choices,default=str(AgentType.random))
    parser.add_argument('--agent_b',type=str,choices=agent_choices,default=str(AgentType.random))
    parser.add_argument('--agent_a_path',type=str,default=None)
    parser.add_argument('--agent_b_path',type=str,default=None)
    parser.add_argument('--turn_delay','-d',type=int,default=0)
    parser.add_argument('--train_episodes',type=int,default=1000)
    parser.add_argument('--eval_freq',type=int,default=10)
    parser.add_argument('-v',action='count',default=0,help="Verbose")
    return parser.parse_args()

def get_player(env, agent_type_str:str, path:str) -> ag.MadAgent_v0:
    if agent_type_str == str(AgentType.human):
        raise ValueError("Human player not yet implemented")
    elif agent_type_str == str(AgentType.random):
        agent = RandomAgent(env.observation_size, env.action_size)
    elif agent_type_str == str(AgentType.dqn):
        agent = DQNAgent(env.observation_size, env.action_size)

    agent.initialize()
    return agent

def printif(*args, flag=True):
    if flag:
        print(*args, flush=True)


def main():
    args = parse_args()
    observer = Observer()

    env = gym.make("mad-v0")
    env.set_config_path(args.env_conf)
    observations = env.reset()
    done = False

    # Currently setup for two AI playing against each other
    # Seperate AI for each player
    agent_a = get_player(env, args.agent_a, args.agent_a_path)
    agent_b = get_player(env, args.agent_b, args.agent_b_path)

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
        if args.v>=1 and not is_burn_in_episode:
            printif("Episode completed: [" + str(episode) + "/" + str(total_episodes) + "]", flag=True)
        agent_a.report_new_episode()
        agent_b.report_new_episode()

        printif(f"Game Over! {info['winner']} won!",flag=PRINT)


    observer.print_final_stats()

if __name__ == "__main__":
    main()
