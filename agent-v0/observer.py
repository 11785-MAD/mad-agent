import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from gym_mad.envs.mad_env_v0 import MadEnv_v0, MadState_v0, MadAction_v0

class Observer:
    """
    Class for observing the entire game state.
    """
    def __init__(self, episode_plot_period=100):
        self.epsiode_plot_period = episode_plot_period
        self.mad_turns = []
        self.turns_acquired_nukes = []
        self.avg_turns_acquired_nukes = 0
        self.avg_mad_turns = 0
        self.episodes = 0
        self.mad_episodes = 0 # mad_episode := episode when both players had nukes at same time

        # per episode records
        self.actions_A = [] # list of action indices
        self.actions_B = []
        self.states = [] # list of state vectors
        self.rewards_A = []
        self.rewards_B = []


    def report_turn(self, S:MadState_v0, A:np.ndarray, R, info, done):
        action = MadAction_v0(A)
        if info['player'] == MadEnv_v0.agent_a:
            self.actions_A.append(np.argmax(action.data))
            self.rewards_A.append(R)
        else:
            self.actions_B.append(np.argmax(action.data))
            self.rewards_B.append(R)
        self.states.append(S.data)

        if done:
            self.episode_finisher()

    def episode_finisher(self):
        """
        """
        self.states = np.array(self.states).T # 10*T
        nuke_indices = [MadState_v0.idx_has_nukes_a, MadState_v0.idx_has_nukes_b]
        nuke_indicator = self.states[nuke_indices,:].sum(axis = 0) # when this is 2, both have nukes
        turns_with_nukes = np.where(nuke_indicator==2)[0]
        if len(turns_with_nukes)>0:
            turn_acquired_nukes = turns_with_nukes[0]
            self.turns_acquired_nukes.append(turn_acquired_nukes)
            self.mad_turns.append(len(nuke_indicator) - turn_acquired_nukes)
            self.mad_episodes += 1

        self.episode += 1

        # plot stuff
        self.plot_actions_over_episode(show_plot=True)

    def report_episode(self, info, turn_acquired_nukes):
        """
        turn_acquired_nukes: the first turn when BOTH agents have nukes
        mad_turns: number of turns passed after both agents after nukes (NOTE: assumes game ends upon nuke firing)
        """
        num_mad_turns = 0
        if (turn_acquired_nukes != -1):
            num_mad_turns = info['turn_count'] - turn_acquired_nukes

        self.mad_turns.append(num_mad_turns)
        self.turns_acquired_nukes.append(turn_acquired_nukes)
        self.episodes = len(self.turns_acquired_nukes)

    def plot_actions_over_episode(self, show_plot=False):
        turns_A = np.arange(len(self.actions_A))
        turns_B = np.arange(len(self.actions_B))

        fig, ax = plt.subplots()       
        y_tick_pos = np.arange(MadAction_v0.action_size)
        y_tick_labels = MadAction_v0.action_strings
        ax.set_yticks(y_tick_pos, y_tick_labels)
        
        ax.plot(turns_A, self.actions_A, label="Agent A")
        ax.plot(turns_B, self.actions_B, label="Agent B")
        ax.set_xlabel("Turns")
        ax.set_ylabel("Action")
        ax.set_title("Actions Chosen over Time")
        ax.grid()
    
        if (show_plot):
            plt.show()

    def do_analysis(self):
        # only compute MAD stats for MAD episodes
        turns_acquired_nukes_mask = np.array(self.turns_acquired_nukes) != -1
        valid_mad_turns = np.array(self.mad_turns)[turns_acquired_nukes_mask]
        valid_turns_acquired_nukes = np.array(self.turns_acquired_nukes)[turns_acquired_nukes_mask]

        self.avg_turns_acquired_nukes = np.sum(valid_turns_acquired_nukes) / len(valid_turns_acquired_nukes)        
        self.avg_mad_turns = np.sum(valid_mad_turns) / len(valid_mad_turns)

        self.mad_episodes = len(valid_mad_turns)

    def print_final_stats(self):
        self.do_analysis()
        print("Total episodes:", self.episodes)
        print("Total MAD episodes:", self.mad_episodes)
        print("Average turn acquired nukes:", np.mean(self.turns_acquired_nukes))
        print("Average MAD turns:", np.mean(self.mad_turns))