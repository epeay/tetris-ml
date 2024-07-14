# @title ActionFeedback, TetrisBoard, TetrisEnv, TetrisGameRecord, TetrominoPiece, Tetrominos

import tensorflow as tf

debug_log_dir = "/content/drive/MyDrive/tensor-logs/debug-logs/"
tf.debugging.experimental.enable_dump_debug_info(debug_log_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

##################
# Environment Prep
##################
import importlib, sys
# if local_libs not in sys.path:
#     sys.path = local_libs + sys.path
def auto_pip(libraries):
    """ Invokes pip if needed. Saves time if not. """
    import importlib
    try:
        for library in libraries:
            importlib.import_module(library)
    except ImportError:
        !pip install {" ".join(libraries)}
# avoids invoking pip unless we need it
auto_pip(["gymnasium"])
# Pull latest changes from local library
######################
# End environment prep
######################

import gymnasium as gym
from gym import spaces
import numpy as np
import pdb
import time



"""
Episode = One tetris game
"""


class ActionFeedback:
    def __init__(self, valid_action=False):
        # Does this action place the mino on or above the board, and not
        # embedded in a wall, for example.
        self.valid_action = valid_action
        self.is_predict = False

    def __str__(self):
        return f"ActionFeedback(valid_action={self.valid_action}, is_predict={self.is_predict})"

class TetrominoPiece:

    BLOCK = '▆'

    def __init__(self, shape:int, patterns):
        self.shape:int = shape
        self.pattern_list = patterns
        self.pattern = patterns[0]
        self.rot = 0

    def __str__(self) -> str:
        return f"TetrominoPiece(shape={Tetrominos.shape_name(self.shape)}, rot={self.rot*90}, pattern= {self.printable_pattern(oneline=True)})"

    def printable_pattern(self, oneline=False):
        ret = []
        pattern = self.get_pattern()
        for i, row in enumerate(pattern):
            row_str = " ".join([str(c) for c in row])
            ret.append(row_str)

            if not oneline:
                ret.append("\n")
            else:
                if i < len(pattern)-1:
                    ret.append(" / ",)
        ret = "".join(ret).replace('1', TetrominoPiece.BLOCK).replace('0', '_')
        return "".join(ret)

    def to_dict(self):
        return {
            "shape": self.shape,
            "pattern": self.pattern
        }

    def get_pattern(self):
        return self.pattern

    def rotate(self):
        """Rotates IN PLACE, and returns the new pattern"""
        self.rot = (self.rot + 1) % 4
        self.pattern = self.pattern_list[self.rot]
        return self.pattern

    def get_height(self):
        return len(self.get_pattern())

    def get_width(self):
        return max([len(x) for x in self.get_pattern()])

    def get_bottom_offsets(self):
        """
        For each column in the shape, returns the gap between the bottom of
        the shape (across all columns) and the bottom of the shape in that
        column.

        Returned values in the list would expect to contain at least one 0, and
        no values higher than the height of the shape.

        For example, an S piece:
        _ X X
        X X _

        Would have offsets [0, 0, 1] in this current rotation. This method is
        used in determining if a piece will fit at a certain position
        in the board.
        """
        pattern = self.get_pattern()
        # pdb.set_trace()
        ret = [len(pattern)+1 for x in range(len(pattern[0]))]
        # Iterates rows from top, down
        for ri in range(len(pattern)):
            # Given a T shape:
            # X X X
            # _ X _
            # Start with row [X X X] (ri=0, offset=1)
            row = pattern[ri]
            # print(f"Testing row {row} at index {ri}")
            for ci, col in enumerate(row):
                if col == 1:
                    offset = len(pattern) - ri - 1
                    ret[ci] = offset

            # Will return [1, 0, 1] for a T shape

        if max(ret) >= len(pattern):
          print(f"Pattern:")
          print(pattern)
          print(f"Bottom Offsets: {ret}")
          print(f"Shape: {self.shape}")
          raise ValueError("Tetromino pattern has incomplete bottom offsets")

        return ret

    def get_top_offsets(self):
        """
        Returns the height of the shape at each column.

        For example, an S piece:
        _ X X
        X X _

        Would have offsets [1, 2, 2] in this current rotation. This provides
        guidance on how to update the headroom list.

        Ideally we should cache this.
        """
        pattern = self.get_pattern()
        ret = [0 for x in len(pattern[0])]
        for ri, row in enumerate(range(pattern, )):
            for col in pattern[row]:
                if pattern[row][col] == 1:
                    ret[col] = max(ret[col], row)
        return ret


class Tetrominos:
    O = 1
    I = 2
    S = 3
    Z = 4
    T = 5
    J = 6
    L = 7
    DOT = 8
    USCORE = 9

    base_patterns = {
        # X X
        # X X
        O: np.array([[1, 1], [1, 1]]),

        # X X X X
        I: np.array([[1, 1, 1, 1]]),

        # _ X X
        # X X _
        S: np.array([[0, 1, 1], [1, 1, 0]]),
        Z: np.array([[1, 1, 0], [0, 1, 1]]),
        T: np.array([[1, 1, 1], [0, 1, 0]]),
        J: np.array([[1, 0, 0], [1, 1, 1]]),
        L: np.array([[0, 0, 1], [1, 1, 1]]),
        DOT: np.array([[1]]),
        USCORE: np.array([[1,1]])
    }

    # Stores patterns for each tetromino, at each rotation
    cache = {}

    def num_tetrominos():
        return len(Tetrominos.base_patterns.keys())

    @staticmethod
    def shape_name(shape):
        if shape == Tetrominos.O:
            return "O"
        elif shape == Tetrominos.I:
            return "I"
        elif shape == Tetrominos.S:
            return "S"
        elif shape == Tetrominos.Z:
            return "Z"
        elif shape == Tetrominos.T:
            return "T"
        elif shape == Tetrominos.J:
            return "J"
        elif shape == Tetrominos.L:
            return "L"
        elif shape == Tetrominos.DOT:
            return "DOT"                # A 1x1 mino, used for testing
        elif shape == Tetrominos.USCORE:
            return "USCORE"             # A 2x1 mino, used for testing
        else:
            raise ValueError("Invalid shape")

    @staticmethod
    def make(shape):
        """
        shape:
        """
        if not Tetrominos.cache:
            for shape, pattern in Tetrominos.base_patterns.items():
                Tetrominos.cache[shape] = [
                    pattern,
                    np.rot90(pattern),
                    np.rot90(pattern, 2),
                    np.rot90(pattern, 3)
                ]


        if shape not in Tetrominos.base_patterns.keys():
            raise ValueError("Invalid shape")

        return TetrominoPiece(shape, Tetrominos.cache[shape])

Tetrominos.std_bag = [
    Tetrominos.O,
    Tetrominos.I,
    Tetrominos.S,
    Tetrominos.Z,
    Tetrominos.T,
    Tetrominos.J,
    Tetrominos.L
]

class TetrisBoard:

    BLOCK = '▆'

    def __init__(self, matrix, height):
        self.play_height = height
        self.height = len(matrix)
        self.width = len(matrix[0])
        self.board = matrix

    def reset(self):
        self.board.fill(0)
        self.piece = None

    def remove_tetris(self):
        to_delete = []
        for r, row in enumerate(self.board):
            if sum(row) == self.width:
                to_delete.append(r)

        if to_delete:
          self.board = np.delete(self.board, to_delete, axis=0)
          self.board.resize((self.height, self.width))
          # pdb.set_trace()

        return len(to_delete)

    def place_piece(self, piece:TetrominoPiece, logical_coords):
        """
        Places a piece at the specified column. Dynamically calculates correct
        height for the piece.

        piece: a TetrominoPiece object
        logical_coords: The logical row and column for the bottom left
            of the piece's pattern
        """
        pattern = piece.get_pattern()

        lrow = logical_coords[0]
        lcol = logical_coords[1]

        p_height = piece.get_height()

        for r in range(p_height):
            pattern_row = pattern[len(pattern)-1-r]
            board_row = self.board[lrow-1+r]

            for i, c in enumerate(pattern_row):
                # Iff c is 1, push it to the board
                board_row[lcol-1+i] |= c


    def find_logical_BL_placement(self, piece:TetrominoPiece, col):
        """
        Assumes the piece fits on the board, horizontally. The piece WILL fit
        vertically, as there are 4 empty rows at the top of the board, which if
        utilized, trigger game over.

        Returns the logical row and column of the bottom left corner of the
        pattern, such that when placed, the piece will sit flush against existing
        tower parts, and not exceed the max board height.

        Given:
        BOARD       PIECE
        5 _ _ _ _
        4 _ _ _ X
        3 _ _ X X   X X X X
        2 _ X X _
        1 X X X X

        Returns (5, 1)

        Given:
        BOARD       PIECE    COL
        5 _ _ _ _
        4 _ _ _ X
        3 _ _ X X   X X X    1 (lcol 2)
        2 _ X X _     X
        1 X X X X

        Returns (3, 1)

        piece: a TetrominoPiece object
        col: zero-index column to place the 0th column of the piece.
        """

        pattern = piece.get_pattern()
        bottom_offsets = np.array(piece.get_bottom_offsets())
        # TODO don't calculate all bottoms because we don't need them all

        # pdb.set_trace()


        board_heights = np.array(self.get_tops()[col:col+piece.get_width()])

        # Given:
        # BOARD       PIECE
        # 5 _ _ _ _
        # 4 _ _ _ X
        # 3 _ _ X X   X X X X
        # 2 _ X X _
        # 1 X X X X
        # Tops -> [1,2,3,4]
        #
        # The sideways I has bottom offsets [0,0,0,0]
        # Start at min(board_tops)+1 and try to place the piece.
        #
        # If placing on row 2, the piece heights would be [2,2,2,2]g
        # Board heights are [1,2,3,4], so this
        # doesn't clear the board for all columns. Try placing on row 3.
        # [3,3,3,3] > [1,2,3,4] ? False
        # Try row 4... False. Try row 5...
        # [5,5,5,5] > [1,2,3,4] ? True
        # So we place the piece on row 5 (index 4)
        #
        # 5 X X X X
        # 4 _ _ _ X
        # 3 _ _ X X
        # 2 _ X X _
        # 1 X X X X
        # (yes, this is a horrible move)

        p_height = piece.get_height()
        p_width = piece.get_width()
        can_place = False

        # TODO Pick better min test height
        # If there's a very narrow, tall tower, and you're placing a flat I
        # just to the left of it, you'll likely test placement for each level of
        # the tower until the piece clears it.
        for place_row in range(min(board_heights)+1, max(board_heights)+2):
            # In the example, place_row would be 2...3...4...5

            bottom_clears_board = all((bottom_offsets + place_row) > board_heights)
            if bottom_clears_board:
                break

        return (place_row, col+1)

    @staticmethod
    def render_state(board, pattern, bl_coords, color=True):
        board = board.copy()

        # Highlight tiles where the last piece was played
        lrow, lcol = bl_coords

        p_height = len(pattern)
        output = False

        for r in range(p_height):
            pattern_row = pattern[len(pattern)-1-r]
            board_row = board[lrow-1+r]

            for i, c in enumerate(pattern_row):
                # Iff c is 1, push it to the board
                if c == 1:
                    board_row[lcol-1+i] = 2

        print(f"{(len(board) -i) % 10} ", end="")
        for i, row in enumerate(reversed(board)):
            if sum(row) == 0 and not output:
                continue
            else:
                output = True

            for cell in row:
                if cell == 2:
                    print(f"\033[36m{TetrisBoard.BLOCK}\033[0m", end=' ')
                elif cell == 1:
                    print(TetrisBoard.BLOCK, end=' ')
                else:
                    print('_', end=' ')
            print()


    def render(self):
        output = False
        for i, row in enumerate(reversed(self.board)):
            # if sum(row) == 0 and not output:
            #     continue
            # else:
            #     output = True

            output = True

            print(f"{(self.height -i) % 10} ", end="")
            for cell in row:
                if cell == 1:
                    print(TetrisBoard.BLOCK, end=' ')
                else:
                    empty = '_'
                    if (self.height -i) > 20:
                        empty = 'X'

                    print(empty, end=' ')
            print()

        if not output:
            print("<<EMPTY BOARD>>")



    def get_tops(self):
        """
        Gets the height of each column on the board.
        This is gonna be inefficient for now.

        A board with only an I at the left side would return [4, 0, 0, ...]
        """
        tops = [0 for _ in range(self.width)]
        for r, row in enumerate(self.board):
            if sum(row) == 0:
                break

            for col, val in enumerate(row):
                if val == 1:
                    tops[col] = r+1

        return tops


class TetrisGameRecord:
    def __init__(self):
        self.id = None  # Populated later
        self.moves = 0
        self.invalid_moves = 0
        self.lines_cleared = 0
        self.cleared_by_size = {
            1: 0,
            2: 0,
            3: 0,
            4: 0
        }
        self.boards = []
        self.pieces = []
        self.placements = []  # Logical coords of BL corner of piece pattern
        self.rewards = []
        self.outcome = []
        self.cumulative_reward = 0
        self.is_predict = []
        self.episode_start_time = time.monotonic_ns()
        self.episode_end_time = None
        self.duration_ns = None
        self.agent_info = {}
        self.logg = None



class TetrisEnv(gym.Env):
    def __init__(self):
        super(TetrisEnv, self).__init__()
        self.board_height = 20
        self.board_width = 10
        self.current_piece = None
        self.pieces = Tetrominos()
        self.reward_history = deque(maxlen=10)
        self.record = TetrisGameRecord()
        self.piece_bag = Tetrominos.std_bag

        # Indexes  0-19 - The visible playfield
        #         20-23 - Buffer for the next piece to sit above the board
        self.state = np.zeros((self.board_height + 4, self.board_width), dtype=int)

        # Creates a *view* from the larger state
        self.current_piece_rows = self.state[20:24]
        self.board = TetrisBoard(self.state, 20)

        # Action space: tuple (column, rotation)
        self.action_space = spaces.MultiDiscrete([self.board_width, 4])

        self.reset()

    def reset(self):
        self.board.reset()
        self.current_piece = self._get_random_piece()
        self.record = TetrisGameRecord()
        return self._get_board_state()

    def step(self, action):
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


        # If any of the top four rows were used -- Game Over
        if np.any(self.board.board[-4:]):
            # Game Over
            done = True
            reward = -1

            self.record.rewards.append(reward)
            self.record.placements.append(None)
            self.record.cumulative_reward += reward
            self.reward_history.append(reward)

            self.close_episode()

            self.board.render()


            return self._get_board_state(), reward, done, info



        reward = self.board_height - lcoords[0]
        print(f"Reward is {reward} for coords {lcoords}")


        # reward = self._calculate_reward()
        done = False

        self.record.rewards.append(reward)
        self.record.placements.append(lcoords)
        self.record.cumulative_reward += reward
        self.reward_history.append(reward)

        # Huzzah!
        lines_gone = self.board.remove_tetris()
        if lines_gone > 0:
            self.record.lines_cleared += 1
            self.record.cleared_by_size[lines_gone] += 1

        reward += lines_gone * 100

        print(f"Reward is {reward} for coords {lcoords}")
        if lines_gone > 0:
            print(f"AND CLEARING {lines_gone} LINES")
            print("----------------------")

        # Prep for next move
        self.current_piece = self._get_random_piece()
        next_state = self._get_board_state()
        return next_state, reward, done, info


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
        return self.pieces.make(random.choice(self.piece_bag))

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

    def _get_board_state(self):

        self.current_piece.get_pattern()
        # TODO use relative coords
        self.board.place_piece(self.current_piece, (21, 1))

        return self.state[np.newaxis, :, :]




def main():

  # Example usage
  env = TetrisEnv()
  env.piece_bag = [Tetrominos.USCORE]
  state = env.reset()

  done = False
  loop_limit = 2
  loop = 0
  while not done and loop < loop_limit:
      action = env.action_space.sample()  # Random action for demonstration
      next_state, reward, done, info = env.step(action)
      env.board.render()
      print(f"Reward: {reward}, Done: {done}")
      print(info)
      print("----------------------")
      loop += 1

  print(env.record.__dict__)


# main()


######################################################
######################################################
######################################################


# @title DQNAgent, TetrisCNN
#####################
# Agent
#


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from typing import NewType
ModelAction = NewType("ModelAction", tuple[int,int])

class TetrisCNN(nn.Module):
    def __init__(self, input_channels, board_height, board_width, action_dim):
        """
        input_channels: 1
        board_height: 24
        board_width: 10
        action_dim: 40  (10 columns * 4 rotations)
        """
        super(TetrisCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * board_height * board_width, 128)  # Adjust based on input size
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the CNN output
        x = torch.relu(self.fc1(x))
        return self.fc2(x)



import random
from collections import deque
from datetime import datetime



class AgentGameInfo:
    def __init__(self):
        self.agent_episode = None
        self.exploration_rate = None
        self.batch_episode = None
        self.batch_size = None

class ModelCheckpoint(dict):
    def __init__(self):
        self.model_state:dict = None
        self.target_model_state:dict = None
        self.optimizer_state:dict = None
        self.replay_buffer:deque = None
        self.exploration:float = None
        self.episode:int = None

    def __setattr__(self, key, value):
        """Class properties become dict key/value pairs"""
        self[key] = value
        super().__setattr__(key, value)

class DQNAgent:
    def __init__(self, input_channels,
                 board_height,
                 board_width,
                 action_dim,
                 learning_rate=0.001,
                 discount_factor=0.99,
                 exploration_rate=1.0,
                 exploration_decay=0.995,
                 min_exploration_rate=0.01,
                 replay_buffer_size=10000,
                 batch_size=64,
                 log_dir:str=None,
                 load_path:str=None
                 ):
        """
        If log_dir is not specified, no logs will be written.
        """
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.reset_key = None
        self.num_rotations = 4
        # Show board state just before sending to model
        self.see_model_view = False

        self.board_height = board_height
        self.board_width = board_width

        self.model = TetrisCNN(input_channels, board_height, board_width, action_dim)
        self.target_model = TetrisCNN(input_channels, board_height, board_width, action_dim)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.game_records = []

        self.writer = SummaryWriter(log_dir) if log_dir is not None else None


        self.agent_episode_count = 0


    def save_model(self, abspath):
        """
        Saves the model to the specified path. Does not include training
        parameters like exploration decay.
        """
        checkpoint = ModelCheckpoint()
        checkpoint.model_state = self.model.state_dict()
        checkpoint.target_model_state = self.target_model.state_dict()
        checkpoint.optimizer_state = self.optimizer.state_dict()
        checkpoint.replay_buffer = self.replay_buffer
        checkpoint.exploration = self.exploration_rate
        checkpoint.episodek = self.agent_episode_count
        torch.save(checkpoint, abspath)

    def load_model(self, abspath):
        checkpoint:ModelCheckpoint = torch.load(abspath)

        self.model.load_state_dict(checkpoint.model_state)
        self.target_model.load_state_dict(checkpoint.target_model_state)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        self.replay_buffer = checkpoint.replay_buffer
        self.exploration_rate = checkpoint.exploration
        self.agent_episode_count = checkpoint.episode



    def log_game_record(self, game_record:TetrisGameRecord):

        if self.writer is None:
            print("WARNING: Not persisting game logs to disk")

        predict = 0
        guess = 0
        for was_predict in game_record.is_predict:
            if was_predict:
                predict += 1
            else:
                guess += 1

        predict_rate = int(predict / (predict + guess) * 10000) / 100

        r = game_record

        # Wrong place to modify the record object
        r.move_guesses = guess
        r.move_predictions = predict
        r.prediction_rate = predict_rate
        r.invalid_move_pct = r.invalid_moves / (r.moves + r.invalid_moves)
        r.avg_time_per_move = r.duration_ns / r.moves / 1000000000

        print(f"Episode {r.agent_info.batch_episode} of {r.agent_info.batch_size}. Agent run #{r.agent_info.agent_episode}")
        print(f"Moves: {r.moves}")
        print(f"Invalid Moves: {r.invalid_moves}")
        print(f"Lines cleared: {r.lines_cleared}  ({str(r.cleared_by_size)})")
        print(f"Highest Reward: {max(r.rewards)}")
        print(f"Prediction Rate: {predict_rate} ({predict} of {predict+guess})")
        print(f"Duration: {r.duration_ns / 1000000000}")
        print(f"Agent Exploration Rate: {r.agent_info.exploration_rate}")
        if r.loss is not None:
            print(f"Loss {r.loss}")

        episode = r.agent_info.agent_episode

        if not self.writer:
            return

        self.writer.add_scalar('Episode/Total Moves', r.moves, episode)
        self.writer.add_scalar('Episode/% Invalid Moves', r.invalid_moves, episode)
        self.writer.add_scalar('Episode/Lines Cleared', r.lines_cleared, episode)
        self.writer.add_scalar('Episode/Cumulative Reward', r.cumulative_reward, episode)
        self.writer.add_scalar('Episode/Prediction Rate', predict_rate, episode)
        self.writer.add_scalar('Episode/Duration', r.duration_ns / 1000000000, episode)
        self.writer.add_scalar('Episode/Avg Time Per Move', r.avg_time_per_move, episode)

        if r.loss is not None:
            self.writer.add_scalar('Episode/Loss', r.loss, episode)

        # Used to more easily identify runs that don't
        # have many episodes, for culling.
        self.writer.add_scalar('Agent/Episode', episode, episode)

    def save_game_records(self, filename="game_records.json"):
        with open(filename, 'w') as f:
            json.dump([record.__dict__ for record in self.game_records], f)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def guess(self):
        """
        Generates a random action based on the action dimensions.
        """
        col = np.random.choice(self.board_width)
        rotation = np.random.choice(4)
        return (col, rotation)

    def predict(self, state) -> ModelAction:
        if len(state.shape) == 3:
            state = np.expand_dims(state, axis=0)  # Add batch dimension if not present
        state = torch.FloatTensor(state)
        q_values = self.model(state)
        action_index = torch.argmax(q_values).item()
        return (action_index // 4, action_index % 4)

    def choose_action(self, state) -> tuple[ModelAction, bool]:
        if random.random() < self.exploration_rate:
            return self.guess(), False
        else:
            action_index = self.predict(state)
            return action_index, True


    def run(self, env, num_episodes=10, train=True):
        total_rewards = []
        target_update_interval = 10

        for episode in range(num_episodes):
            self.agent_episode_count += 1
            if env.record.moves > 0:
                self.game_records.append(env.record)
            state = env.reset()
            step_count = 0
            total_reward = 0
            done = False
            loss = None

            while not done:
                loss = None

                if self.see_model_view:
                    print("MODEL VIEW")
                    env.board.render()
                    print("---------------------")

                if train:
                    action, is_prediction = self.choose_action(state)
                else:
                    action = self.predict(state)
                    is_prediction = True

                next_state, reward, done, info = env.step(action)
                info.is_predict = is_prediction
                env.record.is_predict.append(is_prediction)
                step_count += 1

                if env.record.moves >= 100:
                    print("Hit move cap")
                    done = True

                if done:
                    env.close_episode()

                if train:
                    self.remember(state, action, reward, next_state, done)
                    loss = self.replay()

                state = next_state
                total_reward += reward

            ainfo = AgentGameInfo()
            ainfo.agent_episode = self.agent_episode_count
            ainfo.loss = loss

            # X of Y for this current execution run of the agent
            # Within the lifecycle of this method execution.
            ainfo.batch_episode = episode + 1 if train else episode
            ainfo.batch_size = num_episodes
            ainfo.exploration_rate = self.exploration_rate if train else 0
            env.record.agent_info = ainfo

            if train:
                self.decay_exploration_rate()


            record: TetrisGameRecord = env.record
            record.loss = loss
            print(f"GAME OVER")
            env.render()
            self.log_game_record(record)
            total_rewards.append(total_reward)

            if train and episode % target_update_interval == 0:
                self.update_target_model()

        return total_rewards


    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # pdb.set_trace()

        minibatch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Ensure all states have consistent shapes
        states = np.array(states)
        next_states = np.array(next_states)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor([a[0] * 4 + a[1] for a in actions])  # Ensure action is within valid range
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        max_next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + (self.discount_factor * max_next_q_values * (1 - dones))

        loss = self.loss_fn(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def decay_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)



######################################################
######################################################
######################################################



import os

# Change either of these values to reset the agent. Otherwise we will try
# to keep the agent across multiple notebook cell runs.
run_comment:str = "CNN-cpu-all-Os-datetime-test" # @param {type:"string"}
persist_logs = False # @param {type:"boolean"}

show_board_before_running_model = False # @param {type:"boolean"}


# has_agent = True
# try:
#     agent
# except NameError:
#     has_agent = False
# else:
#     agent:DQNAgent = agent
#     agent.see_model_view = show_board_before_running_model


# if has_agent:
#     del(agent)


save_test = False



# Initialize Tetris environment
env = TetrisEnv()
env.piece_bag = [Tetrominos.O]

input_channels = 1
board_height = 24   # 20 for playfield, 4 for staging next piece
board_width = 10
action_dim = 40  # 4 rotations * 10 columns



log_dir = None
if persist_logs:
    # YYMMDD-HHMMSS
    current_time = datetime.now().strftime('%y%m%d_%H%M%S')
    log_dir = f'/content/drive/MyDrive/tensor-logs/runs/tetris/{current_time}-{run_comment}'
agent = DQNAgent(input_channels, board_height, board_width, action_dim, log_dir=log_dir)



if save_test:
    agent.run(env, 10, train = True)
    print(f"Replay buffer size: {len(agent.replay_buffer)}")
    print(f"Exploration rate: {agent.exploration_rate}")
    print(f"Agent episode count: {agent.agent_episode_count}")
    agent.save_model("/content/drive/MyDrive/tensor-logs/models/test.pth")

else:


    print(f"Replay buffer size: {len(agent.replay_buffer)}")
    print(f"Exploration rate: {agent.exploration_rate}")
    print(f"Agent episode count: {agent.agent_episode_count}")
    print("--------------------------")
    print("loading model")
    agent.load_model("/content/drive/MyDrive/tensor-logs/models/test.pth")
    print(f"Replay buffer size: {len(agent.replay_buffer)}")
    print(f"Exploration rate: {agent.exploration_rate}")
    print(f"Agent episode count: {agent.agent_episode_count}")

sys.exit()





print(len(agent.replay_buffer))
agent.load_model("/content/drive/MyDrive/tensor-logs/models/test.pth")
sys.exit()



num_episodes = 10
target_update_interval = 10

training_tracker = []

model_save_dir = "/content/drive/MyDrive/tensor-logs/models"


def keep_training(agent):
    """
    Run hueristics around recent game performance, and continue
    training if necessary.
    """

    if agent.agent_episode_count >= 10000:
        return False

    records = agent.game_records[-50:]

    avg_line_clears_per_game = np.average([r.lines_cleared for r in records])
    avg_moves_per_game = np.average([r.moves for r in records])
    avg_invalid_move_pct = np.average([r.invalid_move_pct for r in records])

    criteria = []
    criteria.append(avg_line_clears_per_game > 10)
    criteria.append(avg_moves_per_game > 80)
    criteria.append(avg_invalid_move_pct < 1)

    if np.all(criteria):
        return False


    # Resetting progress
    if agent.exploration_rate < 0.1:
        agent.exploration_rate = 0.8

    return True


# agent.run(env, 5, train = True)
# agent.save_model(f"{model_save_dir}/test.pth")

# agent.load_model(f"{model_save_dir}/test.pth")

sys.exit()

save_interval = 500

# while keep_training(agent):
#    rewards = agent.run(env, 50, train = True)

#    if agent.agent_episode_count % save_interval == 0:
#         filename = f"tetris_ep{agent.agent_episode_count}.pth"
#         full_path = os.path.join(model_save_dir, filename)
#         torch.save(agent.model.state_dict(), full_path)
#         print(f"Saved model to {full_path}")


if keep_training(agent):
    print("Training FAILED")
else:
    print("Training SUCCEEDED!!!!!")


filename = f"tetris_ep{agent.agent_episode_count}_TRAINED.pth"
full_path = os.path.join(model_save_dir, filename)
torch.save(agent.model.state_dict(), full_path)
print(f"Saved model to {full_path}")




# agent.train(env, 10)
