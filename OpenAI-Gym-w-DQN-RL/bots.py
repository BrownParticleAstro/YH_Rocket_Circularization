#!/usr/bin/python
from GoT_problem import *
from GoT_types import CellType
import getch
import time

import numpy as np
import gymnasium as gym
import gym.spaces as spaces

from adversarialsearchproblem import AdversarialSearchProblem, GameState
from boardprinter import BoardPrinter
from GoT_types import CellType
import random
import numpy as np
import math

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

import traceback
import copy
import sys
from sys import maxsize as inf

# Custom Environment using provided functions to build a OpenAI Gym Env
class GoT(gym.Env):

    observation_space = gym.spaces.Box(low=1, high=9, shape=(13, 13), dtype=np.float32)
    action_space = gym.spaces.Discrete(4)

    game = GoTProblem("/Users/benbradley/CS Classwork/CS1410/final-project-bpbradle copy/final-project/maps/small_room.txt", 0, message_print=False) 
    internal_board = game._start_state  
    external_board = None
    currentplayer = 0 
    tmp_rounds = 0

    def convert_to_floats(self):
        board = np.array(self.internal_board.board)
        shape = board.shape

        num_arr = np.zeros(shape=shape)
        num_arr += (board == '#').astype(int)
        num_arr += (board == 'X').astype(int) * 2
        num_arr += (board == '1').astype(int) * 3
        num_arr += (board == '*').astype(int) * 4
        num_arr += (board == ' ').astype(int) * 5
        num_arr += (board == 'W').astype(int) * 6
        num_arr += (board == '.').astype(int) * 7
        num_arr += (board == '2').astype(int) * 8
        num_arr += (board == 'O').astype(int) * 9

        self.external_board = num_arr
    
    def reset(self, seed=None, options=None):
        map_string = "/Users/benbradley/CS Classwork/CS1410/final-project-bpbradle copy/final-project/maps/small_room.txt"
        self.game = GoTProblem(map_string, 0, message_print=False) 
        self.internal_board = self.game._start_state
        self.convert_to_floats()
        self.currentplayer = 0
        return self.external_board, {}
    
    def step(self, action):
        if action==0: action = U
        if action==1: action = D
        if action==2: action = L
        if action==3: action = R

        available_actions = self.game.get_available_actions(self.internal_board)
        if not action in available_actions:
            action = list(available_actions)[0]

        result_internal_board = self.game.transition_runner(self.internal_board, action)
        self.game.set_start_state(result_internal_board)
        self.internal_board = result_internal_board
        
        self.tmp_rounds += 1
        # Decide results at max round
        if self.tmp_rounds == (2*500) and not (self.game.is_terminal_state(self.internal_board)):
            player_spaces = GoTProblem._count_space_players(self.internal_board.board, self.internal_board.prev_cell_type, self.game.players)
            if player_spaces[0] > player_spaces[1]:
                self.internal_board.player_locs[1] = None       # Player 0 wins
                if self.currentplayer==0: reward = 100_000
                else: reward = -100_000
            elif player_spaces[0] < player_spaces[1]:
                self.internal_board.player_locs[0] = None       # Player 1 wins
                if self.currentplayer==1: reward = 100_000
                else: reward = -100_000
            else:
                # same size of space, a random player wins
                winner = np.random.randint(0,2)
                self.internal_board.player_locs[1 - winner] = None
            done = True

        # Decide results at terminal state
        if self.game.is_terminal_state(self.internal_board):
            values = self.game.evaluate_state(self.game.get_start_state())
            if values[self.currentplayer] == 1: reward = 100_000
            else: reward = -100_000
            done = True

        # Decide results at midpoints
        else:
            board_array = np.array(self.internal_board.board)
            s_count = np.sum(board_array == 'X')
            b_count = np.sum(board_array == 'O')

            reward = s_count - b_count
            done = False

        self.convert_to_floats()
        self.currentplayer = 1 - self.currentplayer

        # print(f"Action {action} resulted in board...")
        # print(self.internal_board.board)

        return self.external_board, reward, True, False, {}
    
    def render(self, mode="human"):
        return np.array(self.external_board)

# Throughout this file, ASP means adversarial search problem.
class StudentBot:
    """ Write your student bot here """

    turns_elapsed = 0
    rounds_elapsed = 0
    memory = {}

    def heuristic_func(self, state):
        board_array = np.array(state.board)
        s_count = np.sum(board_array == 'X')

        return s_count

    def alpha_beta_cutoff(self, asp, state: GameState, alpha, beta, currplayer, cutoff_ply, d):
            if asp.is_terminal_state(state):
                if state.player_locs[0] == None: score = -inf+1
                else: score = inf-1
                return (None, score, 1, [[]])

            if cutoff_ply == 0: 
                score = self.heuristic_func(state)
                return (None, score, 1, [])
            
            if currplayer==0: 
                node_count = 1
                best_path = []
                maxEval = (None, -inf) 
                actions =  ['R', 'D', 'L', 'U']

                for action in actions:
                    """
                    if self.turns_elapsed==15:
                        print(f"depth: {d}, checking action: {action}")
                        print(state.board)
                    """
                    if node_count >= 5_000:
                        break
                    child = asp.transition_runner(state, action) 
                    eval = self.alpha_beta_cutoff(asp, child, alpha, beta, (currplayer+1)%2, cutoff_ply-1, d+1) # get: Value
                    score = eval[1]
                    node_count += eval[2]

                    if eval[3]!=[]: best_path.append(eval[3])

                    if maxEval[1] < score: maxEval = (action, score)
                    alpha = max(alpha, score)
                    if beta <= alpha:
                        break
                
                path = [[cutoff_ply, currplayer, maxEval[0], maxEval[1]], best_path]
                return (maxEval[0], maxEval[1], node_count, path)
            
            else:
                node_count = 1
                best_path = []
                minEval = (None, inf)
                actions =  ['R', 'D', 'L', 'U']
                
                for action in actions:
                    """
                    if self.turns_elapsed==15:
                        print(f"depth: {d}, checking action: {action}")
                        print(state.board)
                    """
                    if node_count > 5_000:
                        break
                    child = asp.transition_runner(state, action)
                    eval = self.alpha_beta_cutoff(asp, child, alpha, beta, (currplayer+1)%2, cutoff_ply-1, d+1) # get: Value
                    score = eval[1]
                    node_count += eval[2]

                    if eval[3]!=[]: best_path.append(eval[3])

                    if minEval[1] > score: minEval = (action, score)
                    beta = min(beta, score)
                    if beta <= alpha:
                        break
                
                path = [[cutoff_ply, currplayer, minEval[0], minEval[1]], best_path]
                return (minEval[0], minEval[1], node_count, path)

    def decide(self, asp):
        """
        Input: asp, a GoTProblem
        Output: A direction in {'U','D','L','R'}
        To get started, you can get the current
        state by calling asp.get_start_state()
        """
        initial_state = asp.get_start_state()
        player = initial_state.player_to_move()
        self.turns_elapsed += 1
        # print("TURNS ELAPSED : ", self.turns_elapsed)
        start_time = time.time()
        move, move_scores, node_count, path = self.alpha_beta_cutoff(asp, initial_state, -inf, inf, player, 7, 0)
        end_time = time.time()
        time_taken = round(end_time-start_time, 5)
        time_per_node = round(time_taken/node_count, 5)
        print(f"Turn {self.turns_elapsed}: Nodes checked: {node_count}, taking {time_taken} s total or {time_per_node} s per node and took action {move} ")
        path = str(path)
        path = path.replace("[[", "\n[[")
        # print("PATH TAKEN -> ", path)
        return move

    def cleanup(self):
        """
        Input: None
        Output: None
        This function will be called in between
        games during grading. You can use it
        to reset any variables your bot uses during the game
        (for example, you could use this function to reset a
        turns_elapsed counter to zero). If you don't need it,
        feel free to leave it as "pass"
        """
        self.turns_elapsed = 0
        self.rounds_elapsed += 1
        print("ROUNDS ELAPSED : ", self.rounds_elapsed)

class RandBot:
    """Moves in a random (safe) direction"""

    def decide(self, asp):
        """
        Input: asp, a GoTProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(GoTProblem.get_safe_actions(board, loc, ptm))
        if possibilities:
            return random.choice(possibilities)
        return "U"

    def cleanup(self):
        pass

class ManualBot:
    """Bot which can be manually controlled using W, A, S, D"""

    def decide(self, asp: GoTProblem):
        """
        Input: asp, a GoTProblem
        Output: A direction in {'U','D','L','R'}
        """
        # maps keyboard input to {U, D, L, R}
        dir_map = {'A': 'L', 'W': 'U', 
                   'a': 'L', 'w': 'U', 
                   'S': 'D', 'D': 'R', 
                   's': 'D', 'd': 'R'}
        # Command for mac/unix:
        direction = getch.getch()
        # Command for Windows:
        # direction = msvcrt.getch().decode('ASCII')
        return dir_map[direction]

    def cleanup(self):
        pass

class AttackBot:
    """Aggressive bot which attacks opposing player when possible"""

    def __init__(self):
        self.prev_cell_type = None
        self.last_move = None
        self.ptm = None
        self.perm_cell = None
        self.temp_cell = None
        self.opp_temp_cell = None

    def cleanup(self):
        """
        This function is called between
        games. It resets all variables AttackBot uses during the game.        
        """
        self.prev_cell_type = None
        self.last_move = None
        self.ptm = None
        self.perm_cell = None
        self.temp_cell = None
        self.opp_temp_cell = None

    def dist_from_opp(self, opp_loc, ptm_loc):
        """
        Input: other bot's location, bot to move's location
        Output: the shortest distance between the two bots on the board
        """
        dist = 0
        for i in range(len(opp_loc)):
            dist += abs(opp_loc[i] - ptm_loc[i])
        return dist

    def min_dist_to_temp(self, board, ptm_loc):
        locs = self.temp_barrier_locs_from_board(board)
        min_dist = math.inf
        for loc in locs:
            this_dist = self.dist_from_opp(loc, ptm_loc)
            if this_dist < min_dist:
                min_dist = this_dist
        return min_dist

    def temp_barrier_locs_from_board(self, board):
        if self.opp_temp_cell is None:
            return []
        loc_dict = {}
        num_temp = 0
        for r in range(len(board)):
            for c in range(len(board[r])):
                char = board[r][c]
                if char == self.opp_temp_cell:
                    loc_dict[num_temp] = (r, c)
                    num_temp += 1
        loc_list = []
        for index in range(num_temp):
            loc_list.append(loc_dict[index])
        return loc_list

    def decide(self, asp):
        """
        Input: asp, a GoTProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(GoTProblem.get_safe_actions(board, loc, ptm))
        opp_loc = locs[(ptm + 1) % 2]
        if self.ptm is None:
            self.ptm = ptm
            self.perm_cell = CellType.TWO_PERM
            self.temp_cell = CellType.TWO_TEMP
            self.opp_temp_cell = CellType.ONE_TEMP
            if ptm == 0:
                self.perm_cell = CellType.ONE_PERM
                self.temp_cell = CellType.ONE_TEMP
                self.opp_temp_cell = CellType.TWO_TEMP

        if not possibilities:
            return "U"

        if self.prev_cell_type is None:
            "Attack bot starting"
            self.prev_cell_type = self.temp_cell
            this_move = possibilities[0]
            self.last_move = this_move
            return this_move

        # if player needs to return to perm area
        must_return_to_perm = False
        if self.prev_cell_type == self.temp_cell:
            this_move = None
            if self.last_move == "U":
                this_move = "D"
            elif self.last_move == "D":
                this_move = "U"
            elif self.last_move == "R":
                this_move = "L"
            elif self.last_move == "L":
                this_move = "R"
            else:
                raise Exception

            self.prev_cell_type = self.perm_cell
            self.last_move = this_move
            must_return_to_perm = True

        # else, player is (potentially) leaving perm area
        min_dist = math.inf
        min_dist_to_temp = math.inf
        go_for_temp = False
        decision = possibilities[0]
        min_next_loc = [None] * 2
        for move in possibilities:
            next_loc = GoTProblem.move(loc, move)
            dist_from_opponent = self.dist_from_opp(next_loc, opp_loc)
            this_dist_to_temp = self.min_dist_to_temp(board, next_loc)
            if this_dist_to_temp == 0:
                return move

            if not must_return_to_perm:
                # If we are close to temp barrier
                if this_dist_to_temp <= 5 or go_for_temp:
                    go_for_temp = True
                    if this_dist_to_temp < min_dist_to_temp:
                        min_dist_to_temp = this_dist_to_temp
                        decision = move
                        min_next_loc = next_loc

                elif dist_from_opponent < min_dist:
                    min_dist = dist_from_opponent
                    decision = move
                    min_next_loc = next_loc
                    min_dist_to_temp = this_dist_to_temp

                elif dist_from_opponent == min_dist:
                    if this_dist_to_temp < min_dist_to_temp:
                        min_dist_to_temp = this_dist_to_temp
                        decision = move
                        min_next_loc = next_loc

        if not must_return_to_perm:
            if board[min_next_loc[0]][min_next_loc[1]] == self.perm_cell:
                self.prev_cell_type = self.perm_cell
            else:
                self.prev_cell_type = self.temp_cell
            self.last_move = decision
        return self.last_move


class SafeBot:
    """Bot that plays safe and takes area"""

    def __init__(self):
        """
        Sets a bot to its default values for the first game.
        """
        self.prev_move = None
        self.to_empty = []
        self.algo_path = []
        self.path = []
        self.calc_empty = False
        self.order = {"U": ("L", "R"), 
                    "D": ("L", "R"), 
                    "L": ("U", "D"),
                    "R": ("U", "D")}

    def cleanup(self): 
        """
        This function is called between
        games. It resets all variables SafeBot uses during the game.        
        """
        self.prev_move = None
        self.to_empty = []
        self.algo_path = []
        self.path = []
        self.calc_empty = False
        self.order = {"U": ("L", "R"), 
                    "D": ("L", "R"), 
                    "L": ("U", "D"),
                    "R": ("U", "D")}
    
    def get_safe_neighbors_wall(self, board, loc):
        """
        TODO: Nico Add Comment
        """
        neighbors = [
                ((loc[0] + 1, loc[1]), D),
                ((loc[0] - 1, loc[1]), U),
                ((loc[0], loc[1] + 1), R),
                ((loc[0], loc[1] - 1), L),
            ]
        return list(filter(lambda m: board[m[0][0]][m[0][1]] != CellType.WALL, neighbors))

    def get_safe_neighbors_no_wall(self, board, loc, wall):
        """
        TODO: Nico Add Comment
        """
        neighbors = [
                ((loc[0] + 1, loc[1]), D),
                ((loc[0] - 1, loc[1]), U),
                ((loc[0], loc[1] + 1), R),
                ((loc[0], loc[1] - 1), L),
            ]
        return list(filter(lambda m: board[m[0][0]][m[0][1]] != CellType.WALL and board[m[0][0]][m[0][1]] != wall, neighbors))

    def decide(self, asp: GoTProblem):
        """
        Input: asp, a GoTProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        if not self.path:
            if self.calc_empty:
                self.gen_path_to_empty(state)
                self.path += self.to_empty
                self.to_empty = []
                self.calc_empty = False
            else:
                self.gen_space_grab(state)
                self.path += self.algo_path
                self.algo_path = []
                self.calc_empty = True
        move = self.path.pop(0)
        self.prev_move = move
        return move  
        
    def gen_space_grab(self, state : GoTState):
        """
        TODO: Nico Add Comment
        """
        board = state.board
        loc = state.player_locs[state.ptm]
        if state.ptm == 0:
            player_wall = CellType.ONE_PERM
        else:
            player_wall = CellType.TWO_PERM
        avail_actions = {U, D, L, R}
        prev = self.prev_move
        if prev:
            avail_actions.remove(prev)
        else:
            safe_actions = self.get_safe_neighbors_wall(board, loc)
            random.shuffle(safe_actions)
            loc, move = safe_actions[0]
            self.algo_path.append(move)
            avail_actions.remove(move)
            prev = move
        while avail_actions:
            safe_moves = self.get_safe_neighbors_no_wall(board, loc, player_wall)
            safe_moves_wall = self.get_safe_neighbors_wall(board,loc)
            if not safe_moves and not safe_moves_wall:
                self.algo_path.append(U)
                return
            random.shuffle(safe_moves)
            random.shuffle(safe_moves_wall)
            use_wall = True
            for loc, move in safe_moves:
                board_val = board[loc[0]][loc[1]]
                if move in self.order[prev] and move in avail_actions and board_val != player_wall:
                    self.algo_path.append(move)
                    avail_actions.remove(move)
                    prev = move
                    use_wall = False
                    break
            if use_wall:
               for loc, move in safe_moves_wall:
                    board_val = board[loc[0]][loc[1]]
                    if move in self.order[prev] and move in avail_actions:
                        self.algo_path.append(move)
                        avail_actions.remove(move)
                        prev = move
                        use_wall = False
                        break 
        return

    def gen_path_to_empty(self, state: GoTState):
        board = state.board
        player_loc = state.player_locs[state.ptm]
        to_check = [(player_loc, None)]
        checked = {(player_loc, None): None}
        while to_check:
            loc, m = to_check.pop(0)
            neighbors = [
                ((loc[0] + 1, loc[1]), D),
                ((loc[0] - 1, loc[1]), U),
                ((loc[0], loc[1] + 1), R),
                ((loc[0], loc[1] - 1), L),
            ]
            random.shuffle(neighbors)
            for move in neighbors:
                x, y = move[0][0], move[0][1]
                board_val = board[x][y]
                if move not in checked and board_val != CellType.WALL:
                    checked[move] = (loc, m)
                    if board_val == ' ':
                        path = []
                        while move[1] is not None:
                            path.append(move[1])
                            move = checked[move]
                        self.to_empty += path
                        return
                    else:
                        to_check.append(move)
        self.to_empty += [U]
        return
