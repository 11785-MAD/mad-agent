import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from gym_mad.envs.mad_env_v1 import MadEnv_v1, MadState_v1, MadAction_v1, MadGameConfig_v1
from agent import one_hot

class Observer:
    """
    Class for observing the entire game state.
    """
    def __init__(self, episode_plot_period=100, C:MadGameConfig_v1=None):
        self.config = C
        self.epsiode_plot_period = episode_plot_period
        self.mad_turns = []
        self.turns_acquired_nukes = []
        self.avg_turns_acquired_nukes = 0
        self.avg_mad_turns = 0
        self.episodes = 0
        self.mad_episodes = 0 # mad_episode := episode when both players had nukes at same time
        self.fig = None
        self.winning_turns = []
        self.winning_actions = []
        self.winning_cash_sizes = []
        self.A_wins = [] # list of bool, 1 if A won
        self.B_wins = []
        self.draws = []

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
            self.episode_finisher(info, S)

    def episode_finisher(self, info, S):
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

        self.winning_turns.append(min(len(self.actions_A), len(self.actions_B)))
        self.A_wins.append(info["winner"] == MadEnv_v1.agent_a)
        self.B_wins.append(info["winner"] == MadEnv_v1.agent_b)
        self.draws.append(info["winner"] == None)
        self.winning_actions.append(info["action"].action_idx)
        if (info["winner"] == MadEnv_v1.agent_a):
            self.winning_cash_sizes.append(S.cash_a)
        elif (info["winner"] == MadEnv_v1.agent_b):
            self.winning_cash_sizes.append(S.cash_b)
        else:
            self.winning_cash_sizes.append(None)

        self.episodes += 1
        
        # plot stuff
        if self.plotting:
            self.print_action_histograms()
            self.plot_stuff(states)

        # clear stuff
        self.actions_A = [] # list of action indices
        self.actions_B = []
        self.states = [] # list of state vectors
        self.rewards_A = []
        self.rewards_B = []
        
    def plot_stuff(self, states_np_array):
        plt.close('all')
        self.fig, ax_arr = plt.subplots(4, 2, sharey=False,figsize=(15,30))  
        self.fig.subplots_adjust(hspace=0.4, wspace=0.2)
        self.plot_actions_over_episode(ax_arr[0,0], ax_arr[0,1])
        self.plot_stats_over_episode(states_np_array, ax_arr[1,0], ax_arr[1,1], ax_arr[2,0])
        self.plot_agent_cum_reward_over_episode(ax_arr[2,1])
        self.plot_reward_all_actions(ax_arr[3,0], ax_arr[3,1])
        
        plt.show()
        
    def plot_actions_over_episode(self, ax1, ax2):
        turns_A = np.arange(len(self.actions_A))
        turns_B = np.arange(len(self.actions_B))

        y_tick_pos = np.arange(MadAction_v1.action_size)
        y_tick_labels = MadAction_v1.action_strings
        ax1.set_yticks(y_tick_pos, y_tick_labels)
        ax2.set_yticks(y_tick_pos, y_tick_labels)
        
        ax1.scatter(turns_A, self.actions_A, label="Agent A")
        ax2.scatter(turns_B, self.actions_B, label="Agent B", c="g")
        # ax2.set_xticks(turns_B, minor=True)
        ax1.set_title("Agent A Actions")
        ax1.grid()

        ax2.set_title("Agent B Actions")
        ax2.grid()
    
    def plot_stats_over_episode(self, states_np_array, ax3, ax4, ax5):
        turns_all = np.arange(len(self.actions_A) + len(self.actions_B))

        cash_a = states_np_array[MadState_v1.idx_cash_a, :]
        cash_b = states_np_array[MadState_v1.idx_cash_b, :]

        military_a = states_np_array[MadState_v1.idx_military_a, :]
        military_b = states_np_array[MadState_v1.idx_military_b, :]

        income_a = states_np_array[MadState_v1.idx_income_a, :]
        income_b = states_np_array[MadState_v1.idx_income_b, :]

        ax3.set_title("Cash over Time")
        ax3.plot(turns_all, cash_a, label="Agent A")
        ax3.plot(turns_all, cash_b, label="Agent B")
        ax3.set_ylabel("Cash")
        ax3.legend()
        ax3.grid()

        ax5.set_title("Income over Time")
        ax5.plot(turns_all, income_a, label="Agent A")
        ax5.plot(turns_all, income_b, label="Agent B")
        ax5.set_ylabel("Income")
        ax5.legend()
        ax5.grid()

        ax4.set_title("Military over Time")
        ax4.plot(turns_all, military_a, label="Agent A")
        ax4.plot(turns_all, military_b, label="Agent B")
        ax4.set_ylabel("Military")
        ax4.legend()
        ax4.grid()

    def plot_agent_cum_reward_over_episode(self, ax6):
        turns_A = np.arange(len(self.actions_A))
        turns_B = np.arange(len(self.actions_B))

        cum_rewards_A = np.cumsum(np.array(self.rewards_A))
        cum_rewards_B = np.cumsum(np.array(self.rewards_B))

        ax6.set_title("Cumulative Rewards over Time")
        ax6.plot(turns_A, cum_rewards_A, label="Agent A")
        ax6.plot(turns_B, cum_rewards_B, label="Agent B")
        ax6.set_ylabel("Cumulative Reward")
        ax6.legend()
        ax6.grid()
        
    def plot_reward_all_actions(self, ax7, ax8):
        A_rewards = np.zeros((MadAction_v1.action_size,len(self.actions_A)))
        B_rewards = np.zeros((MadAction_v1.action_size,len(self.actions_B)))
        S = MadState_v1(self.config)

        for i, data in enumerate(self.states):
            turn_a = i % 2
            
            for A_idx in range(MadAction_v1.action_size):
                A = MadAction_v1(one_hot(MadAction_v1.action_size, A_idx))
                S.data = data.copy()
                if turn_a:
                    reward, done, winner, info = A.apply_dynamics(S, self.config)
                    A_rewards[A_idx,i//2] = reward
                else:
                    S.swap_agents()
                    reward, done, winner, info = A.apply_dynamics(S, self.config)
                    B_rewards[A_idx,(i-1)//2] = reward

        for A_idx in range(MadAction_v1.action_size):
            ax7.plot(np.arange(len(self.actions_A)),A_rewards[A_idx,:],label=MadAction_v1.action_strings[A_idx])
            ax8.plot(np.arange(len(self.actions_B)),B_rewards[A_idx,:],label=MadAction_v1.action_strings[A_idx])

        ax7.set_title(f"{MadEnv_v1.agent_a} Rewards Per Action")
        ax8.set_title(f"{MadEnv_v1.agent_b} Rewards Per Action")
        for ax in [ax7, ax8]:
            ax.set_ylabel("Reward")
            ax.legend()
            ax.grid()

    def print_final_stats(self):
        #self.do_analysis()
        print("Total episodes:", self.episodes)
        print("Total MAD episodes:", self.mad_episodes)
        print("Average turn acquired nukes:", np.mean(self.turns_acquired_nukes))
        print("Average MAD turns:", np.mean(self.mad_turns))
        print("Average winning turn:", np.mean(self.winning_turns))
        print("Agent A win percentage:", np.mean(self.A_wins))
        print("Agent B win percentage:", np.mean(self.B_wins))
        print("Draw percentage:", np.mean(self.draws))
        print("Agent A winning actions:", self.get_action_histogram(np.array(self.winning_actions)[np.array(self.A_wins)]))
        print("Agent B winning actions:", self.get_action_histogram(np.array(self.winning_actions)[np.array(self.B_wins)]))
        print("Agent A average winning economy size:", np.mean(np.array(self.winning_cash_sizes)[np.array(self.A_wins)]))
        print("Agent B average winning economy size:", np.mean(np.array(self.winning_cash_sizes)[np.array(self.B_wins)]))

    def get_action_histogram(self, action_idx_list):
        hist = {}
        for a in MadAction_v1.action_strings_short:
            hist[a] = 0
        for idx in action_idx_list:
            hist[MadAction_v1.action_strings_short[idx]] += 1
        return hist

    def print_action_histograms(self):
        A = self.get_action_histogram(self.actions_A)
        B = self.get_action_histogram(self.actions_B)
        print(f"A ac hist: {A}")
        print(f"B ac hist: {B}")