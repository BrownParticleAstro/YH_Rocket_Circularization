"""
This file runs the game logic of GoT. 
Basic implementation with two players and ww(s) supported.
"""

################################
# IMPORTS                      
################################
from adversarialsearchproblem import AdversarialSearchProblem, GameState
from boardprinter import BoardPrinter
from GoT_types import CellType
import random
import numpy as np

################################
# CONSTANTS
################################
U = "U"
D = "D"
L = "L"
R = "R"

################################
# CLASSES
################################
class GoTState(GameState):
    def __init__(self, board, player_locs, ptm, prev_cell_type, ww_locs):
        self.board = board
        self.player_locs = player_locs
        self.ptm = ptm
        self.prev_cell_type = prev_cell_type
        self.ww_locs = ww_locs

    def player_to_move(self):
        return self.ptm


class GoTProblem(AdversarialSearchProblem):
    def __init__(self, board_file_loc, first_player, message_print):
        board = GoTProblem._board_from_board_file(board_file_loc)
        player_locs = GoTProblem._player_locs_from_board(board)
        prev_cell_type = [CellType.ONE_PERM, CellType.TWO_PERM]
        ww_locs = GoTProblem._ww_locs_from_board(board)

        self.start_player = first_player
        self._start_state = GoTState(board, player_locs, first_player, prev_cell_type, ww_locs)
        # game constants
        self._num_players = len(player_locs)
        self.half_cells = (GoTProblem._count_fillable_space(board) + 1) // 2
        self.players = {
            0: {"TEMP": CellType.ONE_TEMP, "PERM": CellType.ONE_PERM},
            1: {"TEMP": CellType.TWO_TEMP, "PERM": CellType.TWO_PERM}
        }
        self.message_print = message_print

        self.ww_directions = [[1] * 2] * len(ww_locs)

    def get_available_actions(self, state):
        """
        Returns all moves (even moves that would result in immediate collisions)
        Use get_safe_actions if you want all moves that won't be an immediate collision
        We assume that the player to move is never on the edges of the map.
        All pre-made maps are surrounded by walls to validate this assumption.
        """
        return {U, D, L, R}

    def transition(self, state, action):
        """
        Handles transition from the current state to the next state through the application of a particular player action.
        """
        assert not (self.is_terminal_state(state))
        assert action in self.get_available_actions(state)

        # prepare parts of result state
        ptm = state.ptm
        board = [[elt for elt in row] for row in state.board]
        rows, cols = len(board), len(board[0])
        player_locs = [loc for loc in state.player_locs]
        ww_locs = [loc for loc in state.ww_locs]
        prev_cell_type = [cell for cell in state.prev_cell_type]
        next_ptm = self.get_next_player(ptm)

        r0, c0 = state.player_locs[ptm]

        # get target location after moving
        r1, c1 = GoTProblem.move((r0, c0), action)

        # End of Game: Hit wall / white walker or own trail, the current player loses
        if board[r1][c1] == CellType.WALL or board[r1][c1] == CellType.WHITE_WALKER:
            player_locs[ptm] = None         # end of game
            self.mprint("Player " + self.get_player_head(ptm) + " hit " + board[r1][c1] + " and crashed!")
        elif self._check_hit_own_trail(board, [r1, c1], ptm):
            player_locs[ptm] = None         # end of game
            self.mprint("Player " + self.get_player_head(ptm) + " hit its own tail and crashed!")
        # End of Game: Attack the other player, the current player wins
        elif self._check_hit_other_trail(board, [r1, c1], ptm):
            player_locs[next_ptm] = None
            self.mprint("Player " + self.get_player_head(ptm) +
                  " killed Player" + self.get_player_head(next_ptm) + "'s tail!")

        # Enters SPACE area
        elif board[r1][c1] == CellType.SPACE:
            if prev_cell_type[ptm] == self.players[ptm]['PERM']:
                board[r0][c0] = self.players[ptm]['PERM']
            else:
                board[r0][c0] = self.players[ptm]['TEMP']

            GoTProblem._move_player_and_update(board, ptm, player_locs, r1, c1)
            prev_cell_type[ptm] = self.players[ptm]['TEMP']

        # Enters opponent's perm area
        elif board[r1][c1] == self.players[next_ptm]['PERM']:
            # board[r0][c0] = self.players[ptm]['TEMP'] ######OLD
            """NEW"""
            if prev_cell_type[ptm] == self.players[ptm]['PERM']:
                board[r0][c0] = self.players[ptm]['PERM']
            else:
                board[r0][c0] = self.players[ptm]['TEMP']
            """^^^"""

            GoTProblem._move_player_and_update(board, ptm, player_locs, r1, c1)
            prev_cell_type[ptm] = self.players[ptm]['TEMP']

        # Enters perm area of current player
        else:
            if prev_cell_type[ptm] == self.players[ptm]['TEMP']:
            # The current player returns to PERM from TEMP, trigger claiming and filling
            # The temporary location is in the same fully connected component
            # The player has to return to the base; only reaching the last step of trail is not 'close'
                board[r0][c0] = self.players[ptm]['TEMP']
                enclose_space = self._detect_space_inside(board, ptm)
                capture_ww_list, capture_other_player_bool  = GoTProblem._capture_others(
                    board, enclose_space, ptm, player_locs)

                if capture_other_player_bool:
                    board[player_locs[next_ptm][0]][player_locs[next_ptm][1]] = CellType.DEATH
                    player_locs[next_ptm] = None
                    self.mprint("Player " + self.get_player_head(ptm) + " captured and killed Player" + 
                          self.get_player_head(next_ptm) + "!")
                    return GoTState(board, player_locs, next_ptm, prev_cell_type, ww_locs)
                
                elif capture_ww_list:
                    self.mprint("ww(s) captured at location(s):" + str(*capture_ww_list))
                    for itm_loc in capture_ww_list:
                        assert len(ww_locs)
                        if not GoTProblem._is_same_loc(ww_locs[-1], itm_loc):
                            this_pos = ww_locs.index(itm_loc)
                            ww_locs[this_pos] = ww_locs[-1].copy()
                        ww_locs = ww_locs[:-1]

                space_to_fill = [[i, j] for j in range(cols) for i in range(rows) \
                    if enclose_space[i][j] or board[i][j] == self.players[ptm]["TEMP"]] 

                self.fill_board(board, space_to_fill, ptm)
                prev_cell_type[ptm] = self.players[ptm]['PERM']
            
            else:
                board[r0][c0] = self.players[ptm]['PERM']
                prev_cell_type[ptm] = self.players[ptm]['PERM']

            GoTProblem._move_player_and_update(board, ptm, player_locs, r1, c1)
            # only calculate the space once one claims some new spaces (otherwise won't increase)
            player_spaces = GoTProblem._count_space_players(board, prev_cell_type, self.players)
            space_ptm = player_spaces[ptm] 
            if space_ptm >= self.half_cells:
                player_locs[next_ptm] = None
                self.mprint("Player " + self.get_player_head(ptm) + " won by claiming over half space!")
        
        return GoTState(board, player_locs, next_ptm, prev_cell_type, ww_locs)

    def transition_runner(self, state, action):

        state = self.transition(state, action)
        if state.player_locs[0] == None or state.player_locs[1] == None:
            return state

        ptm = self.get_next_player(state.ptm)  # should not switch
        next_ptm = self.get_next_player(ptm)

        if ptm != self.start_player:
            _player_locs = state.player_locs.copy()

            self.move_ww(state.board, state.player_locs, state.ww_locs)

            if state.player_locs[0] == None and state.player_locs[1] == None:
                state.player_locs = _player_locs
                self._ending_by_space(state.board, state.player_locs, state.prev_cell_type)

        return GoTState(state.board, state.player_locs, next_ptm, state.prev_cell_type, state.ww_locs)

    def move_ww(self, board, player_locs, ww_locs):
        """
        This function handles the logic of white walker movement including trajectory and collisions
        """
        num_wws = len(ww_locs)
        for ww in range(num_wws):

            old_trajectory = self.ww_directions[ww]
            this_loc = ww_locs[ww]

            num_dir_changes = 0
            v_cell_y, v_cell_x = this_loc[0], this_loc[1] + old_trajectory[1]
            v_cell = board[v_cell_y][v_cell_x]  # cell above or below the current ww
            h_cell_y, h_cell_x = this_loc[0] + old_trajectory[0], this_loc[1]
            h_cell = board[h_cell_y][h_cell_x]  # cell next to the current ww
            diag_cell_y, diag_cell_x = this_loc[0] + old_trajectory[0], this_loc[1] + old_trajectory[1]
            diag_cell = board[diag_cell_y][diag_cell_x]
            # now see if ww trajectory should be changed due to hitting a wall...
            new_trajectory = old_trajectory.copy()

            if v_cell in CellType.STOPS_WHITE_WALKERS:  # ... a wall above or below the ww
                new_trajectory[1] *= -1
                num_dir_changes += 1
            if h_cell in CellType.STOPS_WHITE_WALKERS:  # ... a wall next to the ww
                new_trajectory[0] *= -1
                num_dir_changes += 1
            # see if ww trajectory should be changed due to hitting a wall (NOT CORNER)
            if (diag_cell in CellType.STOPS_WHITE_WALKERS and num_dir_changes == 0) or (
            diag_cell_y, diag_cell_x) in ww_locs[:ww + 1]:
                new_trajectory[0] *= -1
                new_trajectory[1] *= -1

            self.ww_directions[ww] = new_trajectory

            # now, we actually move the ww
            new_loc = (this_loc[0] + new_trajectory[0], this_loc[1] + new_trajectory[1])
            cell = board[new_loc[0]][new_loc[1]]

            if cell == CellType.SPACE:
                board[new_loc[0]][new_loc[1]] = CellType.WHITE_WALKER
                board[this_loc[0]][this_loc[1]] = CellType.SPACE

                ww_locs[ww] = new_loc

            # see if the ww killed a player
            elif cell == CellType.ONE_TEMP or new_loc == player_locs[0]:  # did we kill player 1
                player_locs[0] = None
                self.mprint("White Walker " + str(ww) + " killed player 1!")
            elif cell == CellType.TWO_TEMP or new_loc == player_locs[1]:  # did we kill player 2
                player_locs[1] = None
                self.mprint("White Walker " + str(ww) + " killed player 2!")
            else:  
                # if the ww can't go in its previous or opposite directory, we'll move the ww up, down,
                self.mprint("White Walker " + str(ww) +
                            " is trapped. It won't move this round!.")

    ##############################################
    # EDGE CASE CHECKINGS TO CALL BEFORE EACH MOVE
    ##############################################
    def _check_hit_own_trail(self, board, loc, ptm):
        """If the player hit one's own trail, the current player loses"""
        r, c = loc
        if board[r][c] == self.players[ptm]["TEMP"]:
            return True
        return False

    def _check_hit_other_trail(self, board, loc, ptm):
        """If the player hits the other player's trail, the opponent loses"""
        r, c = loc
        next_ptm = 1 - ptm
        if board[r][c] == self.players[next_ptm]["TEMP"]:
            return True
        if board[r][c] == self.get_player_head(next_ptm):
            return True
        return False

    def is_terminal_state(self, state):
        """
        This method checks to see if the input state is the terminal state.
        """
        num_players_left = 0
        for pl in state.player_locs:
            if not (pl == None):
                num_players_left += 1

        return num_players_left == 1

    def intercept_max_rounds(self, state):
        """
        This method is called when the max number of game rounds has been achieved and the game must end.
        """
        print("Reached maximum running steps, the game is forced to quit...")
        self._ending_by_space(state.board, state.player_locs, state.prev_cell_type)

    def _ending_by_space(self, board, player_locs, prev_cell_type):
        """
        This method is called in order to determine who won the game based on which player has more space claimed.
        """
        # if not self.is_terminal_state(state):
        player_spaces = GoTProblem._count_space_players(board, prev_cell_type, self.players)
        if player_spaces[0] > player_spaces[1]:
            player_locs[1] = None
            self.mprint("Player 1 claimed more space!")
        elif player_spaces[0] < player_spaces[1]:
            player_locs[0] = None
            self.mprint("Player 2 claimed more space!")
        else:
            # same size of space, a random player wins
            winner = np.random.randint(0,2)
            player_locs[1 - winner] = None
            self.mprint(f"Same space claimed, randomly pick a lucky winner...")

    def evaluate_state(self, state):
        """
        Note that, since players take turns sequentially,
        ties are impossible.
        """
        assert self.is_terminal_state(state)

        values = [0.0 if pl == None else 1 for pl in state.player_locs]
        return values

    ###### STATIC METHODS FOR IMPLEMENTING METHODS ABOVE ######

    ##############################################
    # CLAIMING SPACES ONCE TRIGGERED - CLOSE TRAIL
    ##############################################

    def _detect_space_inside(self, board, ptm):
        """get the unclaimed region surrounded by the claimed space and the trail
        (the trail is not set 1 in the returned mask matrix)
        Logic: there are three kinds of regions (components) on the board:
            Region 1. Enclosed area (might of area = 0, or capture others) to claim
            Region 2. A + a: the claimed space and the trail
            Region 3. Outside area

        :return enclose_space,
            a 2d list with area enclosed by the trail and PERM territories set to 1
        """
        rows, cols = len(board), len(board[0])

        def _is_border(_r, _c):
            # Check whether location [_r, _c] is on the trail or previously claimed space
            return board[_r][_c] == self.players[ptm]['PERM'] or board[_r][_c] == self.players[ptm]['TEMP']

        def _loc_on_board(_r, _c):
            # Check whether the location [_r, _c] is valid on the board (index out of bounds)
            if _r < 0 or _c < 0:
                return False
            if _r >= rows or _c >= cols:
                return False
            return True

        neighbours = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, 1], [1, -1], [-1, 1]]
        # Region 2 is tagged as 2 initally
        enclose_space = [[1 if not _is_border(i, j) else 2 for j in range(cols)] for i in range(rows)]
        queue = []

        # BFS - get region 3, tagged as 0
        queue.append([0, 0])
        enclose_space[0][0] = 0

        while queue:
            tmp_loc = queue.pop(0)
            for nbr in neighbours:
                r, c = tmp_loc[0] + nbr[0], tmp_loc[1] + nbr[1]
                if not _loc_on_board(r, c):
                    continue
                if enclose_space[r][c] == 1:
                    enclose_space[r][c] = 0
                    queue.append([r, c])

        # get the inner space tagged as 1 (region 1), others 0 (region 2 and region 3)
        enclose_space = [[enclose_space[i][j] % 2 for j in range(cols)] for i in range(rows)]

        return enclose_space

    def fill_board(self, board, space_to_fill, ptm):
        """Run this function to claim space once the trail is closed"""
        # space to fill does contain the temporary trail
        while space_to_fill:
            row, col = space_to_fill.pop()
            board[row][col] = self.players[ptm]['PERM']

    def mprint(self, message):
        if self.message_print:
            print(message)

    ###############################################
    # STATIC METHODS FOR IMPLEMENTING METHODS ABOVE
    ###############################################
    @staticmethod
    def _count_space_players(board, prev_cell_type, players):
        """Calculate the permanent spaces taken by each player. Use np for fast computation"""
        unique, counts = np.unique(board, return_counts=True)
        space_one = np.where(unique == players[0]['PERM'])
        space_two = np.where(unique == players[1]['PERM'])
        space_count = [1 if prev_cell_type[0] == players[0]['PERM'] else 0, 
                       1 if prev_cell_type[1] == players[1]['PERM'] else 0]
        if len(space_one[0]):
            space_count[0] += counts[space_one][0]
        if len(space_two[0]):
            space_count[1] += counts[space_two][0]
        return space_count

    @staticmethod
    def _capture_others(board, mask_space, ptm, player_locs):
        """
        Decide whether the newly claimed space encloses any ww or the other player.
        :param list mask_space, the region to detect white walkers or opponent will be tagged 1
        :param int ptm, the current player id
        """
        rows, cols = len(board), len(board[0])
        loc_next_r, loc_next_c = player_locs[GoTProblem.get_next_player(ptm)]

        capture_ww_list = []
        capture_other_player_bool = bool(mask_space[loc_next_r][loc_next_c])
        for i in range(rows):
            for j in range(cols):
                if mask_space[i][j] and board[i][j] == CellType.WHITE_WALKER:
                    capture_ww_list.append([i, j])

        return capture_ww_list, capture_other_player_bool

    @staticmethod
    def _move_player_and_update(board, ptm, player_locs, r1, c1):
        """
        adds player location to map, then stores the player location in player_locs
        """
        board[r1][c1] = GoTProblem.get_player_head(ptm)  # add player location to map
        player_locs[ptm] = (r1, c1)  # stores player location

    @staticmethod
    def _board_from_board_file(board_file_loc):
        board_file = open(board_file_loc)
        board = []
        for line in board_file.readlines():
            line = line.strip()
            row = [

                random.choice(CellType.powerup_list) if c == "?" else c

                for c in line
                if not (c == "\n")
            ]
            board.append(row)
        return board
    
    @staticmethod
    def get_next_player(ptm):
        return 1 - ptm

    @staticmethod
    def _count_fillable_space(board):
        rows, cols = len(board), len(board[0])
        assert rows, cols
        total_cells = rows * cols
        for i in range(rows):
            for j in range(cols):
                if board[i][j] == CellType.WALL or board[i][j] == CellType.WHITE_WALKER:
                    total_cells -= 1
        return total_cells

    @staticmethod
    def _player_locs_from_board(board):
        loc_dict = {}
        for r in range(len(board)):
            for c in range(len(board[r])):
                char = board[r][c]

                if GoTProblem._is_int(char):
                    index = int(char) - 1
                    loc_dict[index] = (r, c)

        loc_list = []
        num_players = len(loc_dict)
        for index in range(num_players):
            loc_list.append(loc_dict[index])
        return loc_list

    @staticmethod
    def _ww_locs_from_board(board):
        loc_dict = {}
        num_wws = 0
        for r in range(len(board)):
            for c in range(len(board[r])):
                char = board[r][c]
                if char == CellType.WHITE_WALKER:
                    loc_dict[num_wws] = (r, c)
                    num_wws += 1

        loc_list = []
        for index in range(num_wws):
            loc_list.append(loc_dict[index])

        return loc_list

    @staticmethod
    def _is_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def move(loc, direction):
        """
        Produces the location attained by going in the given direction
        from the given location.
        loc will be a (<row>, <column>) double, and direction will be
        U, L, D, or R.
        """
        r0, c0 = loc
        if direction == U:
            return (r0 - 1, c0)
        elif direction == D:
            return (r0 + 1, c0)
        elif direction == L:
            return (r0, c0 - 1)
        elif direction == R:
            return (r0, c0 + 1)
        else:
            raise ValueError("The input direction is not valid.")
    
    @staticmethod
    def _is_same_loc(loc1, loc2):
        assert len(loc1) == 2 and len(loc2) == 2
        return loc1[0] == loc2[0] and loc1[1] == loc2[1]

    ###### HELPFUL FUNCTIONS FOR YOU ######

    @staticmethod
    def is_cell_player(board, loc):
        """
        Input:
            board- a list of lists of characters representing cells
            loc- location (<row>, <column>) on the board
        Output:
            Returns true if the cell at loc is a player, which is true when
            the player is a digit, or false otherwise.
        """
        r, c = loc
        return board[r][c].isdigit()

    @staticmethod
    def get_safe_actions(board, loc, ptm):
        """
        Given a game board and a location on that board,
        returns the set of actions that don't result in immediate collisions.
        Input:
            board- a list of lists of characters representing cells
            loc- location (<row>, <column>) to find safe actions from
            has_shield- boolean for whether the player has shield or not
        Output:
            returns the set of actions that don't result in immediate collisions.
            An immediate collision occurs when you run into a barrier, wall, or
            the other player
        """
        if ptm == 0:
            unsafe_vals = {CellType.WALL, CellType.ONE_TEMP, '2'}
        elif ptm == 1:
            unsafe_vals = {CellType.WALL, CellType.TWO_TEMP, '1'}
        else:
            raise Exception

        safe = set()
        for action in {U, D, L, R}:
            r1, c1 = GoTProblem.move(loc, action)

            if board[r1][c1] not in unsafe_vals:
                safe.add(action)
        return safe
    
    @staticmethod
    def get_player_head(ptm):
        return str(ptm + 1)

    @staticmethod
    def visualize_state(state, colored):
        print(BoardPrinter.state_to_string(state, colored))
