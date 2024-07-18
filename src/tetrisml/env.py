import datetime
import gymnasium as gym
import numpy as np
import time
import random

from collections import deque
from datetime import datetime
from gymnasium import spaces

from .tetrominos import Tetrominos, TetrominoPiece
from .minos import MinoPlacement, MinoShape
from .board import TetrisBoard
from .logging import TetrisGameRecord



class ActionFeedback:
    def __init__(self, valid_action=False):
        # Does this action place the mino on or above the board, and not
        # embedded in a wall, for example.
        self.valid_action = valid_action
        self.is_predict = False

    def __str__(self):
        return f"ActionFeedback(valid_action={self.valid_action}, is_predict={self.is_predict})"



class TetrisEnv(gym.Env):

    def __init__(self, piece_bag=None, board_height=20, board_width=10):
        super(TetrisEnv, self).__init__()
        self.board_height = board_height
        self.board_width = board_width
        self.current_piece = None
        self.pieces = Tetrominos()
        self.reward_history = deque(maxlen=10)
        self.record = TetrisGameRecord()
        self.piece_bag = Tetrominos.std_bag if piece_bag is None else piece_bag
        self.step_history:list[MinoPlacement] = []
        self.random:random.Random = None
        self.random_seed = None

        # Indexes  0-19 - The visible playfield
        #         20-23 - Buffer for the next piece to sit above the board
        self.state = np.zeros((self.board_height + 4, self.board_width), dtype=int)

        # Creates a *view* from the larger state
        # of all but the top four rows
        self.current_piece_rows = self.state[:-4]
        self.board = TetrisBoard(self.state, self.board_height - 4)

        # Action space: tuple (column, rotation)
        self.action_space = spaces.MultiDiscrete([self.board_width, 4])

        self.reset()



    def reset(self, seed:int=None):
        self.board.reset()

        if seed is not None:
            self.random_seed = seed
        else:
            ts = datetime.now().timestamp()
            self.random_seed = int(ts * 1000)

        # We need reproducibility
        # This seed only impacts random values generated from self.random.
        # The global random module is not affected.
        self.random = random.Random(self.random_seed)

        self.current_piece = self._get_random_piece()
        self.record = TetrisGameRecord()
        self.step_history:list[MinoPlacement] = []
        return self._get_board_state()

    def step(self, action:tuple[int,int]):
        """
        action: zero-indexed tuple of (column, rotation)
        """
        # ([0-9], [0-3])
        col, rotation = action
        lcol = col + 1

        info = ActionFeedback()

        # Rotate the piece to the desired rotation
        for _ in range(rotation):
            self.current_piece.rotate()  # Rotates IN PLACE

        # Clear the area above the visible board. If this range is used during
        # piece placement, the game is over.
        self.board.board[-4:].fill(0)

        # Check for right-side overflow
        # Given a horizontal I piece on col 0
        # right_lcol would be 4. The piece would occupy lcolumns 1-4.
        right_lcol = lcol-1 + self.current_piece.get_width()
        if right_lcol > self.board_width:
            # Ignore this action and try again.
            #
            # For example, a location is chosen which extends
            # the piece over the edge of the board.
            done = False
            info.valid_action = False
            self.current_piece.rot = 0
            self.record.invalid_moves += 1
            reward = -1

            return self._get_board_state(), reward, done, info

        info.valid_action = True
        lcoords = None

        lcoords = self.board.find_logical_BL_placement(self.current_piece, col)
        self.board.place_piece(self.current_piece, lcoords)

        self.record.moves += 1
        self.record.boards.append(self.board.board.copy())
        self.record.pieces.append(self.current_piece.to_dict())
        self.current_piece.rot = 0


        self.record.placements.append(lcoords)


        # If any of the top four rows were used -- Game Over
        if np.any(self.board.board[-4:]):
            # Game Over
            done = True
            reward = -1

            self.record.rewards.append(reward)
            self.record.cumulative_reward += reward
            self.reward_history.append(reward)

            self.close_episode()

            return self._get_board_state(), reward, done, info

        # reward = self.board_height - lcoords[0]

        reward = self._calculate_reward()
        done = False

        self.record.rewards.append(reward)
        self.record.cumulative_reward += reward
        self.reward_history.append(reward)

        # Huzzah!
        lines_gone = self.board.remove_tetris()
        if lines_gone > 0:
            self.record.lines_cleared += 1
            self.record.cleared_by_size[lines_gone] += 1

        reward += lines_gone * 100

        # Prep for next move
        self.current_piece = self._get_random_piece()
        next_board_state = self._get_board_state()
        return next_board_state, reward, done, info


    def close_episode(self):
        """
        Wraps up episode stats. Public method is available for agent to call
        if needed.
        """

        if self.record.episode_end_time:
            # Already closed
            return

        self.record.episode_end_time = time.monotonic_ns()
        self.record.duration_ns = self.record.episode_end_time - self.record.episode_start_time


    def render(self):
        self.board.render()

    def _get_random_piece(self):
        return self.pieces.make(self.random.choice(self.piece_bag))

    def _is_valid_action(self, piece, lcol):
        piece = self.current_piece

        if lcol < 1 or lcol > self.board_width:
            return False

        # An O piece on col 1 would occupy cols 1-2
        if lcol + piece.get_width() -1 > self.board_width:
            return False
        return True

    def _calculate_reward(self):

        # Evaluate line pack
        # Packed lines produces a higher score
        # Big narrow tower would produce a low score
        active_lines = 0
        board_tiles = 0
        lines_cleared = 0
        board = self.board.board

        for row in self.board.board:
            row_sum = sum(row)
            board_tiles += row_sum
            if row_sum == 0:
                continue

            active_lines += 1
            if row_sum == self.board.width:
                lines_cleared += 1

        if active_lines == 0:
            return 0

        # Simulating an extra 10 packed tiles per line cleared
        line_clear_bonus = 10

        # Narrow towers get lower rewards. Starting with row 3, every row that
        # has < 50% pack, gets a penalty of this many additional empty tiles when
        # calculating the pack score.
        sharp_tower_penalty = 3
        sharp_tower_pack_min = 0.5
        underpacked_lines = 0

        line_pack_pct = [(sum(x) / self.board.width) for x in self.board.board]
        high_tower_penalty = 0
        for pct in line_pack_pct[3:active_lines]:
            if pct < sharp_tower_pack_min:
                underpacked_lines += 1

        high_tower_penalty = underpacked_lines * sharp_tower_penalty

        line_score = (board_tiles+(10*lines_cleared)) / float(self.board_width * active_lines + high_tower_penalty)

        line_pack_pct = [sum(x) / self.board.width for x in self.board.board]

        reward = line_score  # That's all for now
        return reward

    def _get_board_state(self) -> np.ndarray:
        """
        Returns a copy of the board state. Shape is (1, H, W)
        """
        # Copy the board state
        # INEFFICIENT!!!
        # TODO Do I still need to do this?
        state = self.board.board.copy()
        return state[np.newaxis, :, :]
    



def find_possible_moves(env:TetrisEnv, s:MinoShape):
    """
    Given a mino and a board, returns a list of possible moves for the mino.
    """

    board:TetrisBoard = env.board

    options = []
    tower_heights = np.array(board.get_tops())

    # This is gonna be inefficient for now

    for c in range(board.width - s.width + 1):
        lcoords = board.find_logical_BL_placement(s.get_piece(), c)

        # Assume placement of Shape:
        # Field:      | Shape:
        # X O O O     |    O O O
        # X X O       |      O
        # X X X       |
        # As shown at coords (2,2)
        #
        # Board height is               [3, 2, 1, 0, ...]
        # Shape height (from bottom) is    [1, 0, 1]

        # [1, 0, 1]
        mino_col_heights = np.array(s.get_bottom_gaps())
        # [2, 1, 0]
        tower_col_heights = tower_heights[c:c+s.width]

        # The gap between the bottom of the mino and the tower
        # [0, 0, 2]
        gaps = mino_col_heights - 1 + lcoords[0] - tower_col_heights

        piece:TetrominoPiece = s.get_piece()
        backup_rows = board.place_shape(s, lcoords)
        reward = env._calculate_reward()

        # Revert the board
        for r in range(len(backup_rows)):
            board.board[lcoords[0]-1+r] = backup_rows[r]

        placement = MinoPlacement(s, lcoords, gaps.tolist(), reward)
        options.append(placement)

    return options


