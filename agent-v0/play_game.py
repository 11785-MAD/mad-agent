#!/usr/bin/env python3

import sys
import time
import numpy as np
import argparse as ap
from enum import Enum, auto

import gym
import gym_mad
from agent import RandomAgent
from dqn import DQNAgent

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
    return parser.parse_args()

def get_player(env, agent_type_str, path):
    if agent_type_str == str(AgentType.human):
        raise ValueError("Human player not yet implemented")
    elif agent_type_str == str(AgentType.random):
        agent = RandomAgent(env.observation_size, env.action_size)
    elif agent_type_str == str(AgentType.dqn):
        agent = DQNAgent(env.observation_size, env.action_size)

    agent.initialize(env)

def printif(*args, flag=True):
    if flag:
        print(*args)


def main():
    args = parse_args()

    env = gym.make("mad-v0")
    env.set_config_path(args.conf)
    observations = env.reset()
    done = False

    # Currently setup for two AI playing against each other
    # Seperate AI for each player
    agent_a = get_player(env, args.agent_a, args.agent_a_path)
    agent_b = get_player(env, args.agent_b, args.agent_b_path)

    check_nukes = True # check if both players have nukes

    turn_acquired_nukes = -1

    episode = 0
    is_burn_in_episode = False
    total_episodes = (train_episodes * (eval_freq+1)) // eval_freq

    while episode < total_episodes:

        is_burn_in_episode = False
        if agent_a.is_burning_in or agent_b.is_burning_in:
            episode = 0
            is_burn_in_episode = True

        
        while not done:

            printif("------------------------------------")
            printif(f"Begin {env.current_player}'s turn")
            if env.current_player == env.agent_a:
                action_vec = agent_a(observations[env.agent_a])
                new_observations, reward, done, info = env.step(action_vec)
                printif(f"Player took action {info['action'].action_str}")
                printif(info["turn_desc"])
                printif("New State:")
                env.render()
                agent_a.report_SARS(
                    observations[env.agent_a],
                    action_vec, reward, new_observations[env.agent_a], done)
            else:
                action_vec = agent_b(observations[env.agent_b])
                new_observations, reward, done, info = env.step(action_vec)
                printif(f"Player took action {info['action'].action_str}")
                printif(info["turn_desc"])
                printif("New State:")
                env.render()
                agent_b.report_SARS(
                    observations[env.agent_b],
                    action_vec, reward, new_observations[env.agent_b], done)

            players_have_nukes = env.check_both_nukes()
            if (check_nukes and players_have_nukes):
                turn_acquired_nukes = info["turn_count"]
                check_nukes = False

            time.sleep(args.turn_delay)
            observations = new_observations

            printif(f"End turn {info['turn_count']}\n")

        # end while not done
        agent_a.report_new_episode()
        agent_b.report_new_episode()

        printif(f"Game Over! {info['winner']} won!")

        num_mad_turns = 0
        if (turn_acquired_nukes != -1):
            num_mad_turns = info['turn_count'] - turn_acquired_nukes
        print("Turn acquired nukes: " + str(turn_acquired_nukes) + ", num_mad_turns: " + str(num_mad_turns))


if __name__ == "__main__":
    main()
