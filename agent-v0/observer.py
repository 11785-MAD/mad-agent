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
        self.fig = None

        # per episode records
        self.actions_A = [] # list of action indices
        self.actions_B = []
        self.states = [] # list of state vectors
        self.rewards_A = []
        self.rewards_B = []
        self.plotting = False


    def report_turn(self, S:MadState_v0, A:np.ndarray, R, info, done):
        action = MadAction_v0(A)
        if info['player'] == MadEnv_v0.agent_a:
            self.actions_A.append(np.argmax(action.data))
            self.rewards_A.append(R)
        else:
            self.actions_B.append(np.argmax(action.data))
            self.rewards_B.append(R)
        self.states.append(S.data.copy())

        if done:
            self.episode_finisher()

    def episode_finisher(self):
        """
        """
        states = np.array(self.states).T # 10*T
        nuke_indices = [MadState_v0.idx_has_nukes_a, MadState_v0.idx_has_nukes_b]
        nuke_indicator = states[nuke_indices,:].sum(axis = 0) # when this is 2, both have nukes
        turns_with_nukes = np.where(nuke_indicator==2)[0]
        if len(turns_with_nukes)>0:
            turn_acquired_nukes = turns_with_nukes[0]
            self.turns_acquired_nukes.append(turn_acquired_nukes)
            self.mad_turns.append(len(nuke_indicator) - turn_acquired_nukes)
            self.mad_episodes += 1

        self.episodes += 1
        
        # plot stuff
        if self.plotting:
            self.plot_actions_over_episode()

        # clear stuff
        self.actions_A = [] # list of action indices
        self.actions_B = []
        self.states = [] # list of state vectors
        self.rewards_A = []
        self.rewards_B = []

    def plot_actions_over_episode(self):
        turns_A = np.arange(len(self.actions_A))
        turns_B = np.arange(len(self.actions_B))

        if self.fig is not None:
            self.fig.close()
        self.fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False,figsize=(15,15))      
        y_tick_pos = np.arange(MadAction_v0.action_size)
        y_tick_labels = MadAction_v0.action_strings
        ax1.set_yticks(y_tick_pos, y_tick_labels)
        ax2.set_yticks(y_tick_pos, y_tick_labels)
        
        ax1.scatter(turns_A, self.actions_A, label="Agent A")
        ax2.scatter(turns_B, self.actions_B, label="Agent B", c="g")
        ax2.set_xlabel("Turns")
        ax1.set_title("Agent A Actions")
        ax1.grid()

        ax2.set_title("Agent B Actions")
        ax2.grid()
    
        plt.show()

    def print_final_stats(self):
        #self.do_analysis()
        print("Total episodes:", self.episodes)
        print("Total MAD episodes:", self.mad_episodes)
        print("Average turn acquired nukes:", np.mean(self.turns_acquired_nukes))
        print("Average MAD turns:", np.mean(self.mad_turns))