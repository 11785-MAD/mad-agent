#!/usr/bin/env python3

import gym
import gym_mad
import numpy as np
from agent import MadAgent_v0
import time


def main(args):

    env = gym.make("mad-v0")
    observations = env.reset()
    done = False

    # Currently setup for two AI playing against each other
    # Seperate AI for each player
    agent_a = MadAgent_v0(env.observation_size, env.action_size)
    agent_b = MadAgent_v0(env.observation_size, env.action_size)

    while not done:

        print(f"Begin {env.current_player}'s turn")
        if env.current_player == env.agent_a:
            action_vec = agent_a(observations[env.agent_a])
            new_observations, reward, done, info = env.step(action_vec)
            print(f"Player took action {info['action'].action_str}")
            print("New State:")
            env.render()
            agent_a.report_SARS(
                observations[env.agent_a],
                action_vec, reward, new_observations[env.agent_a])
        else:
            action_vec = agent_b(observations[env.agent_b])
            new_observations, reward, done, info = env.step(action_vec)
            print(f"Player took action {info['action'].action_str}")
            print("New State:")
            env.render()
            agent_b.report_SARS(
                observations[env.agent_b],
                action_vec, reward, new_observations[env.agent_b])

        time.sleep(3)
        observations = new_observations

    print("Game Over!")


if __name__ == "__main__":
    # TODO: Do we need commandline args?
    main(None)
