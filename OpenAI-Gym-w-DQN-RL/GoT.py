# import os
# import subprocess
# os.system("python /Users/benbradley/final-project-bpbradle/final-project/gamerunner.py -bots student safe -map /Users/benbradley/final-project-bpbradle/final-project/maps/empty_room.txt -multi_test 100 -no_image -no_msg")
# os.system("python /Users/benbradley/final-project-bpbradle/final-project/gamerunner.py -bots student safe -map /Users/benbradley/final-project-bpbradle/final-project/maps/small_room.txt -multi_test 100 -no_image -no_msg")
# os.system("python /Users/benbradley/final-project-bpbradle/final-project/gamerunner.py -bots student safe -map /Users/benbradley/final-project-bpbradle/final-project/maps/large_room.txt  -multi_test 100 -no_image -no_msg")

import numpy as np
import gymnasium as gym
import gym.spaces as spaces
import support

from adversarialsearchproblem import AdversarialSearchProblem, GameState
from boardprinter import BoardPrinter
from GoT_types import CellType
import random
import numpy as np
import math

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.adversarial.airl import AIRL
import sb3_contrib
from sb3_contrib import QRDQN

import traceback
import copy
import sys
from sys import maxsize as inf
import itertools as itt

from stable_baselines3.ppo import MlpPolicy
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from gym.envs.registration import register

from GoT_problem import GoTProblem, GoTState

################################
# CONSTANTS
################################
U = "U"
D = "D"
L = "L"
R = "R"

################################
# CUSTOM ENV
################################

class GoT(gym.Env):

    observation_space = gym.spaces.Box(low=1, high=9, shape=(13, 13), dtype=np.float32)
    action_space = gym.spaces.Discrete(4)

    game = GoTProblem("maps/small_room.txt", 0, message_print=False) 
    internal_board = game._start_state  
    external_board = None
    currentplayer = 0 
    tmp_rounds = 0

    def convert_to_floats(self):
        board = np.array(self.internal_board.board)
        shape = board.shape

        num_arr = np.zeros(shape=shape)
        num_arr += (board == '#').astype(int) * 9
        num_arr += (board == 'X').astype(int) * 7
        num_arr += (board == '1').astype(int) * 8
        num_arr += (board == '*').astype(int) * 5
        num_arr += (board == ' ').astype(int) * 1
        num_arr += (board == 'W').astype(int) * 4
        num_arr += (board == '.').astype(int) * 3
        num_arr += (board == '2').astype(int) * 6
        num_arr += (board == 'O').astype(int) * 2

        self.external_board = num_arr
    
    def reset(self, seed=None, options=None):
        map_string = "maps/small_room.txt"
        self.game = GoTProblem(map_string, 0, message_print=False) 
        self.internal_board = self.game._start_state
        self.convert_to_floats()
        self.currentplayer = 0
        self.tmp_rounds = 0
        return self.external_board
    
    # You have an unfinished board
      # Play action, if it's done return evaluate_state()
      # If not, enemy plays action, if it's done return evaluate_state()
      # If not done still, return 

    def step(self, action):
        if action==0: action = U
        if action==1: action = D
        if action==2: action = L
        if action==3: action = R

        # Run transition with inputted action
        result_internal_board = self.game.transition_runner(self.internal_board, action)
        self.game.set_start_state(result_internal_board)
        self.internal_board = result_internal_board
        
        self.tmp_rounds += 1
        # Decide results at max round
        if self.tmp_rounds == (2*500) and not (self.game.is_terminal_state(self.internal_board)):
            player_spaces = GoTProblem._count_space_players(self.internal_board.board, self.internal_board.prev_cell_type, self.game.players)
            if player_spaces[0] > player_spaces[1]:
                self.internal_board.player_locs[1] = None       # Player 0 wins
                if self.currentplayer==0: reward = 10
                else: reward = -10
            elif player_spaces[0] < player_spaces[1]:
                self.internal_board.player_locs[0] = None       # Player 1 wins
                if self.currentplayer==1: reward = 10
                else: reward = -10
            else:
                # same size of space, a random player wins
                winner = np.random.randint(0,2)
                self.internal_board.player_locs[1 - winner] = None
            done = True

        # Decide results at terminal state
        elif self.game.is_terminal_state(self.internal_board):
            values = self.game.evaluate_state(self.game.get_start_state())
            if values[self.currentplayer] == 1: reward = 10
            else: reward = -10
            done = True

        # Decide results at midpoints
        else:
            board_array = np.array(self.internal_board.board)
            s_count = np.sum(board_array == 'X')
            b_count = np.sum(board_array == 'O')

            reward = 0

            self.currentplayer = 1 - self.currentplayer
            bot_action = support.bots.RandBot().decide(self.game)
            result_internal_board = self.game.transition_runner(self.internal_board, bot_action)
            self.game.set_start_state(result_internal_board)
            self.internal_board = result_internal_board
            if self.game.is_terminal_state(self.internal_board):
                values = self.game.evaluate_state(self.game.get_start_state())
                if values[self.currentplayer] == 1: reward = -10
                else: reward = 10
                done = True
            else:
                done = False

        #if done:
            #print(f"Action {action} resulted in board...")
            #print(self.internal_board.board)


        self.convert_to_floats()
        self.currentplayer = 1 - self.currentplayer

        return self.external_board, reward, done
    
    def render(self, mode="human"):
        return np.array(self.external_board)
    
# ------------------ #
# ------------------ #
# ------------------ #
# ------------------ #
# ------------------ #

""" env = GoT()

l1 = [1, 2, 3]
l2 = ['A', 'B', 'C']
l3 = ['d', 'e', 'f']
l4 = ['word1', 'word2', 'word3']
l5 = [10, 20, 30]

result = []
for n in range(1, 5):
    for ls in itt.combinations([l1, l2, l3, l4, l5], n):
        result+= list(itt.product(*ls))

top_set = (None, -inf)
for setup in result:
    _ = _
    _ = _

    model = QRDQN("MlpPolicy", env, verbose=1)
    learner_rewards_before_training, _ = evaluate_policy(
        model, env, 100, return_episode_rewards=True,
    )
    model.learn(total_timesteps=200_000, progress_bar=True)
    learner_rewards_after_training, _ = evaluate_policy(
        model, env, 100, return_episode_rewards=True,
    )

    print(f"mean reward before training with setup {setup} :", np.mean(learner_rewards_before_training))
    print(f"mean reward after training with setup {setup} :", np.mean(learner_rewards_after_training))

    if learner_rewards_after_training > top_set[1]:
        model.save("dqn_small_model")
        top_set = (setup, learner_rewards_after_training) """