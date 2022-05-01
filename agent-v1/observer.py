import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from gym_mad.envs.mad_env_v1 import MadEnv_v1, MadState_v1, MadAction_v1

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


    def report_turn(self, S:MadState_v1, A:np.ndarray, R, info, done):
        action = MadAction_v1(A)
        if info['player'] == MadEnv_v1.agent_a:
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
        nuke_indices = [MadState_v1.idx_has_nukes_a, MadState_v1.idx_has_nukes_b]
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
            self.print_action_histograms()
            self.plot_actions_over_episode()
            self.plot_stats_over_episode(states)
            self.plot_reward_over_episode()

        # clear stuff
        self.actions_A = [] # list of action indices
        self.actions_B = []
        self.states = [] # list of state vectors
        self.rewards_A = []
        self.rewards_B = []

    def plot_reward_over_episode(self):
        turns_A = np.arange(len(self.actions_A))
        turns_B = np.arange(len(self.actions_B))

        cum_rewards_A = np.cumsum(np.array(self.rewards_A))
        cum_rewards_B = np.cumsum(np.array(self.rewards_B))

        plt.close('all')
        self.fig, (ax1) = plt.subplots(1, 1, sharey=False,figsize=(15,15))      

        ax1.set_title("Cumulative Rewards over Time")
        ax1.plot(turns_A, cum_rewards_A, label="Agent A")
        ax1.plot(turns_B, cum_rewards_B, label="Agent B")
        ax1.set_ylabel("Cumulative Reward")
        ax1.set_xlabel("Turns")
        ax1.legend()
        ax1.grid()

        plt.show()

    def plot_stats_over_episode(self, states_np_array):
        turns_A = np.arange(len(self.actions_A))
        turns_B = np.arange(len(self.actions_B))
        turns_all = np.arange(len(self.actions_A) + len(self.actions_B))

        cash_a = states_np_array[MadState_v1.idx_cash_a, :]
        cash_b = states_np_array[MadState_v1.idx_cash_b, :]

        military_a = states_np_array[MadState_v1.idx_military_a, :]
        military_b = states_np_array[MadState_v1.idx_military_b, :]

        income_a = states_np_array[MadState_v1.idx_income_a, :]
        income_b = states_np_array[MadState_v1.idx_income_b, :]

        plt.close('all')
        self.fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=False,figsize=(15,15))      

        ax1.set_title("Cash over Time")
        ax1.plot(turns_all, cash_a, label="Agent A")
        ax1.plot(turns_all, cash_b, label="Agent B")
        ax1.set_ylabel("Cash")
        ax1.legend()
        ax1.grid()

        ax2.set_title("Income over Time")
        ax2.plot(turns_all, income_a, label="Agent A")
        ax2.plot(turns_all, income_b, label="Agent B")
        ax2.set_ylabel("Income")
        ax2.legend()
        ax2.grid()

        ax3.set_title("Military over Time")
        ax3.plot(turns_all, military_a, label="Agent A")
        ax3.plot(turns_all, military_b, label="Agent B")
        ax3.set_ylabel("Military")
        ax3.set_xlabel("Turns")
        ax3.legend()
        ax3.grid()

        plt.show()

    def plot_actions_over_episode(self):
        turns_A = np.arange(len(self.actions_A))
        turns_B = np.arange(len(self.actions_B))

        plt.close('all')
        self.fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False,figsize=(15,15))      
        y_tick_pos = np.arange(MadAction_v1.action_size)
        y_tick_labels = MadAction_v1.action_strings
        ax1.set_yticks(y_tick_pos, y_tick_labels)
        ax2.set_yticks(y_tick_pos, y_tick_labels)
        
        ax1.scatter(turns_A, self.actions_A, label="Agent A")
        ax2.scatter(turns_B, self.actions_B, label="Agent B", c="g")
        ax2.set_xlabel("Turns")
        # ax2.set_xticks(turns_B, minor=True)
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

    def get_action_histogram(self):
        A_hist = {}
        B_hist = {}

        for a in MadAction_v1.action_strings_short:
            A_hist[a] = 0
            B_hist[a] = 0

        for idx in self.actions_A:
            A_hist[MadAction_v1.action_strings_short[idx]] += 1

        for idx in self.actions_B:
            B_hist[MadAction_v1.action_strings_short[idx]] += 1

        return A_hist, B_hist

    def print_action_histograms(self):
        A, B = self.get_action_histogram()
        print(f"A ac hist: {A}")
        print(f"B ac hist: {B}")