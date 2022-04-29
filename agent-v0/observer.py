import numpy as np

class Observer:
    """
    Class for observing the entire game state.
    """
    def __init__(self):
        self.mad_turns = []
        self.turns_acquired_nukes = []
        self.avg_turns_acquired_nukes = 0
        self.avg_mad_turns = 0
        self.episodes = 0
        self.mad_episodes = 0 # mad_episode := episode when both players had nukes at same time

    def report_turn(self, S, A, R, Sp, info, done):

        pass

    def report_episode(self, turn_acquired_nukes, mad_turns):
        """
        turn_acquired_nukes: the first turn when BOTH agents have nukes
        """
        self.mad_turns.append(mad_turns)
        self.turns_acquired_nukes.append(turn_acquired_nukes)
        self.episodes = len(self.turns_acquired_nukes)

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
        print("Average turn acquired nukes:", self.avg_turns_acquired_nukes)
        print("Average MAD turns:", self.avg_mad_turns)