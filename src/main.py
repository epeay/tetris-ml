# @title ActionFeedback, TetrisBoard, TetrisEnv, TetrisGameRecord, TetrominoPiece, Tetrominos

# debug_log_dir = "/content/drive/MyDrive/tensor-logs/debug-logs/"
# tf.debugging.experimental.enable_dump_debug_info(debug_log_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)


##################
# Environment Prep
##################
'''
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
'''

import datetime
import gymnasium as gym
import json
import numpy as np
import os
import pdb
import time
import random
import sys
import yaml

from collections import deque
from gymnasium import spaces
from numpy import ndarray as NDArray


WORKSPACE_ROOT = os.path.join(os.path.expanduser("~"), "source", "tetris-ml")

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

# Verify TensorFlow is using CPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


"""
Episode = One tetris game
mino = Tetromino
"""


class TMLConfig(dict):
    def __init__(self):
        self.workspace_dir:str = os.path.normpath(WORKSPACE_ROOT)
        self.storage_root:str = os.path.join(WORKSPACE_ROOT, "storage")
        self.tensorboard_log_dir:str = os.path.join(self.storage_root, "tensor-logs")

    def __setattr__(self, key, value):
        """Class properties become dict key/value pairs"""
        self[key] = value
        super().__setattr__(key, value)

    def __getattr__(self, key):
        return self[key]

config = TMLConfig()

# Load ../config.yaml
config_path = os.path.join(os.getcwd(), "config.yaml")
with open(config_path, 'r') as stream:
    try:
        config.update(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)




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
    def make(shape, rot=0):
        """
        shape:
        """
        if not Tetrominos.cache:
            for shape, pattern in Tetrominos.base_patterns.items():
                Tetrominos.cache[shape] = [
                    np.array(pattern),
                    np.array(np.rot90(pattern)),
                    np.array(np.rot90(pattern, 2)),
                    np.array(np.rot90(pattern, 3))
                ]


        if shape not in Tetrominos.base_patterns.keys():
            raise ValueError("Invalid shape")

        ret = TetrominoPiece(shape, Tetrominos.cache[shape])
        for _ in range(rot):
            ret.rotate()
        return ret




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
        self.pattern_list:list[NDArray] = patterns
        self.pattern:NDArray = patterns[0]
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
            "pattern": self.pattern.tolist(),
        }

    def get_pattern(self):
        return self.pattern

    def get_shape(self, rot):
        """
        Returns the pattern for the specified rotation.

        A 'Z' mino would return [[1,1,0],[0,1,1]] for rot=0
        XX_
        _XX
        """
        return self.pattern_list[rot]

    def rotate(self):
        """Rotates IN PLACE, and returns the new pattern"""
        self.rot = (self.rot + 1) % 4
        self.pattern = self.pattern_list[self.rot]
        return self.pattern

    def get_height(self, rot=None):
        if rot:
            return len(self.get_shape(rot))
        else:
            return len(self.get_pattern())

    def get_width(self, rot=None):
        if rot:
            return len(self.get_shape(rot)[0])
        else:
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


class MinoShape:
    """
    A MinoShape is a single rotation of a Tetromino piece. It is a 2D array.
    Importantly, this class is meant to be immutable.
    """
    def __init__(self, shape_id:int, rot:int):
        self.shape:list[list[int]] = Tetrominos.cache[shape_id][rot]
        self.shape_id:int = shape_id
        self.shape_rot:int = rot
        self.height:int = len(self.shape)
        self.width:int = len(self.shape[0])

        # Private
        self._bottom_gaps:list[int] = None

    def __str__(self):
        return f"MinoShape(shape={self.shape})"

    def to_jsonable(self):
        return {
            "id": self.shape_id,
            "name": Tetrominos.shape_name(self.shape_id),
            "rot": self.shape_rot,
            "shape": self.shape.tolist()
        }

    def get_piece(self)->TetrominoPiece:
        """
        Backtrack to the TetrominoPiece of this shape.
        """
        ret = Tetrominos.make(self.shape_id, self.shape_rot)
        return ret

    def get_bottom_gaps(self):
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

        if self._bottom_gaps:
            return self._bottom_gaps

        pattern = self.shape
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

        self._bottom_gaps = ret

        return self._bottom_gaps



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
        self.board:NDArray = matrix

    def reset(self):
        self.board.fill(0)
        self.piece = None

    def remove_tetris(self):
        to_delete = []
        for r, row in enumerate(self.board):
            if sum(row) == self.width:
                to_delete.append(r)

        if to_delete:

            # TODO Handle this more efficiently
            # I believe BOTH of these operations make copies of the source data
            self.board = np.delete(self.board, to_delete, axis=0)

            # Odd workaround because vscode is messing with the memory management
            # of numpy by holding a reference to the board, which prevents the
            # resize in place.
            refcheck = True  # This is the default
            if "ISDEBUG" in os.environ.keys():
                refcheck = False

            self.board.resize((self.height, self.width), refcheck=refcheck)

        return len(to_delete)


    def place_shape(self, s:MinoShape, logical_coords:tuple[int,int]) -> list[NDArray]:
        piece = s.get_piece()
        return self.place_piece(piece, logical_coords)

    def place_mino(self, mino:TetrominoPiece, rot:int, logical_coords:tuple[int,int]) -> list[NDArray]:
        """
        A simple wrapper around place_piece, knowing that I want to make TetrominoPiece
        stateless.
        """
        old_rot = mino.rot
        mino.rot = rot
        ret = self.place_piece(mino, logical_coords)
        mino.rot = old_rot
        return ret

    def place_piece(self, piece:TetrominoPiece, logical_coords, alt_board=None) -> list[NDArray]:
        """
        Places a piece at the specified column. Dynamically calculates correct
        height for the piece.

        piece: a TetrominoPiece object
        logical_coords: The logical row and column for the bottom left
            of the piece's pattern
        alt_board: An optional board to place the piece on. If not provided,
            the board attribute is used.
        """
        pattern = piece.get_pattern()
        board = self.board if alt_board is None else alt_board

        lrow = logical_coords[0]
        lcol = logical_coords[1]

        lr = logical_coords[0]
        row_backups = [x.copy() for x in board[lr-1:lr-1+piece.get_height()]]

        p_height = piece.get_height()

        for r in range(p_height):
            pattern_row = pattern[len(pattern)-1-r]
            board_row = board[lrow-1+r]

            for i, c in enumerate(pattern_row):
                # Iff c is 1, push it to the board
                board_row[lcol-1+i] |= c

        return row_backups




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
    def render_state(board, highlight_shape:MinoShape=None, highlight_bl_coords=None, color=True):
        board = board.copy() 
        output = False

        if highlight_shape is not None and highlight_bl_coords is not None:
            shape = highlight_shape.shape
            p_height = len(shape)
            lrow = highlight_bl_coords[0]
            lcol = highlight_bl_coords[1]

            for r in range(p_height):
                pattern_row = shape[len(shape)-1-r]
                board_row = board[lrow-1+r]

                for i, c in enumerate(pattern_row):
                    # Iff c is 1, push it to the board
                    if c == 1:
                        board_row[lcol-1+i] = 2

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
        

        print("=======================================")


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
        self.placements = []
        self.rewards = []
        self.outcomes = []
        self.cumulative_reward = 0
        self.is_predict = []
        self.episode_start_time = time.monotonic_ns()
        self.episode_end_time = None
        self.duration_ns = None
        self.agent_info = {}

    def to_jsonable(self):
        # omitting boards
        ret = {
            "id": self.id,
            "moves": self.moves,
            "invalid_moves": self.invalid_moves,
            "lines_cleared": self.lines_cleared,
            "cleared_by_size": self.cleared_by_size,
            "pieces": self.pieces,
            "placements": self.placements,
            "rewards": self.rewards,
            "outcomes": self.outcomes,
            "cumulative_reward": self.cumulative_reward,
            "episode_start_time": self.episode_start_time,
            "episode_end_time": self.episode_end_time,
            "duration_ns": self.duration_ns,
            "agent_info": self.agent_info
        }

        return ret

class TetrisEnv(gym.Env):

    def __init__(self, piece_bag=None):
        super(TetrisEnv, self).__init__()
        self.board_height = 20
        self.board_width = 10
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
        self.current_piece_rows = self.state[20:24]
        self.board = TetrisBoard(self.state, 20)

        # Action space: tuple (column, rotation)
        self.action_space = spaces.MultiDiscrete([self.board_width, 4])

        self.reset()



    def reset(self, seed:int=None):
        self.board.reset()

        if seed is not None:
            self.random_seed = seed

        if self.random_seed is None:
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
        action: tuple of (column, rotation)
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

        reward = self.board_height - lcoords[0]

        # reward = self._calculate_reward()
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
        # Copy the board state
        # INEFFICIENT!!!
        state = self.board.board.copy()

        # This is actually kind of nice. This way the current piece is only
        # visible when generating the state for the model.
        
        # TODO Revisit this
        self.board.place_piece(self.current_piece, (21, 1), state)
        return state[np.newaxis, :, :]


class MinoPlacement():
    def __init__(self,
                 shape:MinoShape,
                 bl_coords:tuple[int, int],
                 gaps_by_col:list[int],
                 reward:float
                 ) -> None:
        self.shape = shape
        self.bl_coords = bl_coords
        self.reward:float = reward

        # At which columns does the piece not sit flush?
        # Field:      | Shape:
        # X O O O     |    O O O
        # X X O       |      O
        # X X X       |
        # The gaps are [0, 0, 2]
        self.gaps:list[int] = gaps_by_col
        self.empty_tiles_created = sum(self.gaps)
        self.is_flush = self.empty_tiles_created == 0

    def to_jsonable(self):
        return {
            "shape": self.shape.to_jsonable(),
            "bl_coords": self.bl_coords,
            "gaps": self.gaps,
            "reward": self.reward,
            "empty_tiles_created": self.empty_tiles_created,
            "is_flush": self.is_flush
        }


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

class GameHistory:
    def __init__(self):
        self.timestamp = datetime.now()
        self.unix_ts = self.timestamp.timestamp()
        self.id = self.make_id()
        self.seed:int = None
        self.bag:list[str] = []
        self.placements:list[MinoPlacement] = []
        # Not including overflow
        self.field_dims:tuple[int,int] = (20, 10)

        # Probably not going to use this, but it's another
        # data collector so let's hold onto it.
        self.record:TetrisGameRecord = None

    def make_id(self):
        """
        Produces IDs like 240714-beeeef
        """
        hexstr = '0123456789abcdef'
        ret = ""
        ret += self.timestamp.strftime("%y%m%d") + "-"
        ret += ''.join([random.choice(hexstr) for x in range(6)])
        return ret

    def to_jsonable(self):
        ret = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "unix_ts": self.unix_ts,
            "seed": self.seed,
            "bag": self.bag,
            "placements": [x.to_jsonable() for x in self.placements],
            "field_dims": self.field_dims,
            "record": self.record.to_jsonable() if self.record is not None else {}
            }

        return ret


import time

def mcts():
    env = TetrisEnv()
    game_logs = []
    file_ts = None


    for _ in range(50):
        env.reset()
        history:GameHistory = GameHistory()
        print(f"Starting game {history.id}")
        history.seed = env.random_seed
        history.bag = env.piece_bag

        move = 0
        while True:
            move += 1
            piece = env.current_piece
            possibilities = []
            # TODO: Some minos don't need four rotations
            for i in range(4):
                possibilities += (find_possible_moves(env, MinoShape(piece.shape, i)))

            possibilities = np.array(possibilities, dtype=MinoPlacement)

            sorted_possibilities = sorted(possibilities, key=lambda x: x.reward - x.empty_tiles_created, reverse=True)
            best_choice:MinoPlacement = sorted_possibilities[0]
            history.placements.append(best_choice)

            env.step((best_choice.bl_coords[1]-1, best_choice.shape.shape_rot))
            print("--------------------------")
            print(f"Move {move}")
            print(f"Best Choice: {best_choice}")
            env.render()
            print("--------------------------")


            if env.board.board[-4:].any():
                env.close_episode()
                history.record = env.record


                print("Game Over")
                print(f"GAME ID: {history.id}")
                print(f"Final Reward: {env.record.cumulative_reward}")
                print(f"Lines Cleared: {env.record.lines_cleared}")
                print(f"Invalid Moves: {env.record.invalid_moves}")
                print(f"Clears by Size: {env.record.cleared_by_size}")
                print(f"Duration: {env.record.duration_ns / 1000000000}")
                print(f"Moves: {env.record.moves}")
                print(f"Game Seed: {env.random_seed}")

                if env.record.cleared_by_size[4] > 0:
                    print("Tetris!!!!!!!!!!")
                    print("Tetris!!!!!!!!!!")
                    print("Tetris!!!!!!!!!!")
                    print("Tetris!!!!!!!!!!")
                    print("Tetris!!!!!!!!!!")
                    # time.sleep(10)

                # time.sleep(3)
                break

        game_logs.append(history)
        if file_ts is None:
            file_ts = history.timestamp.strftime("%y%m%d_%H%M%S")

    save_game_logs(game_logs, f"game_logs_{file_ts}.json")


def save_game_logs(game_logs:list[GameHistory], path:str="game_logs.json"):
    ret = {"games": {}}

    for game in game_logs:
        ret["games"][game.id] = game.to_jsonable()

    with open(path, "w") as outfile:
        json.dump(ret, outfile, indent=4)


# mcts()
# sys.exit()


def main():
  # Example usage
  env = TetrisEnv()
  env.piece_bag = Tetrominos.std_bag
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

################################################################################
################################################################################
################################################################################
################################################################################

# @title DQNAgent, TetrisCNN
#####################
# Agent
#


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cpu")
print(f"Using device {device}")


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

    MODE_UNSET = 0
    MODE_EXPERT_LEARNING = 1

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
        self.mode = DQNAgent.MODE_UNSET

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


    def run(self, env:TetrisEnv, num_episodes=10, train=True, playback_list:list[GameHistory] = None):
        total_rewards = []
        target_update_interval = 10

        playback = None

        if playback_list is not None:
            playback_itr = iter(playback_list)
            playback = next(playback_itr)

        for episode in range(num_episodes):
            self.agent_episode_count += 1
            # Capture the most recent game record.
            # TODO But do we not capture the final record?
            if env.record.moves > 0:
                self.game_records.append(env.record)

            move_list:list[MinoPlacement] = None

            if playback is not None:
                state = env.reset(seed=playback.seed)
                move_list = playback.placements
            else:
                state = env.reset()

            step_count = 0
            total_reward = 0
            done = False
            loss = None

            while not done:
                loss = None

                if train:
                    action, is_prediction = self.choose_action(state)
                else:
                    action = self.predict(state)
                    is_prediction = True


                col, rot = action
                planned_shape = MinoShape(env.current_piece.shape, rot)

                # Apply the action, choose the next piece, etc
                next_state, reward, done, info = env.step(action)

                if info.valid_action:
                    TetrisBoard.render_state(state[0], planned_shape, (21, col+1))
                    TetrisBoard.render_state(next_state[0])
                    print("Next state ^^^")
                    print(f"Sum of state: {np.sum(state)}")
                    print(f"Sum of next state: {np.sum(next_state)}")
                    # time.sleep(0.2)


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

            if train and playback_list is None:
                # If in playback mode, we'll set the exploration rate, later
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




################################################################################
################################################################################
################################################################################
################################################################################

import os

# Change either of these values to reset the agent. Otherwise we will try
# to keep the agent across multiple notebook cell runs.
run_comment:str = "CNN-cpu-all-Os-datetime-test" # @param {type:"string"}
# persist_logs = False # @param {type:"boolean"}
persist_logs = config.persist_logs

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
env = TetrisEnv(piece_bag=[Tetrominos.O])


input_channels = 1
board_height = 24   # 20 for playfield, 4 for staging next piece
board_width = 10
action_dim = 40  # 4 rotations * 10 columns



log_dir = None
if persist_logs:
    # YYMMDD-HHMMSS
    current_time = datetime.now().strftime('%y%m%d_%H%M%S')
    log_dir = os.path.join(config.tensorboard_log_dir, f'{current_time}-{run_comment}')
agent = DQNAgent(input_channels, board_height, board_width, action_dim, log_dir=log_dir)



# if save_test:
#     agent.run(env, 10, train = True)
#     print(f"Replay buffer size: {len(agent.replay_buffer)}")
#     print(f"Exploration rate: {agent.exploration_rate}")
#     print(f"Agent episode count: {agent.agent_episode_count}")
#     agent.save_model("/content/drive/MyDrive/tensor-logs/models/test.pth")

# else:


#     print(f"Replay buffer size: {len(agent.replay_buffer)}")
#     print(f"Exploration rate: {agent.exploration_rate}")
#     print(f"Agent episode count: {agent.agent_episode_count}")
#     print("--------------------------")
#     print("loading model")
#     agent.load_model("/content/drive/MyDrive/tensor-logs/models/test.pth")
#     print(f"Replay buffer size: {len(agent.replay_buffer)}")
#     print(f"Exploration rate: {agent.exploration_rate}")
#     print(f"Agent episode count: {agent.agent_episode_count}")

# sys.exit()

# print(len(agent.replay_buffer))
# agent.load_model("/content/drive/MyDrive/tensor-logs/models/test.pth")
# sys.exit()

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

    print(f"Evaluating hueristics over {len(records)} games")
    print(f"Avg Line Clears: {avg_line_clears_per_game}")
    print(f"Avg Moves: {avg_moves_per_game}")
    print(f"Avg Invalid Move %: {avg_invalid_move_pct}")

    if np.all(criteria):
        return False


    # Resetting progress
    if agent.exploration_rate < 0.1:
        agent.exploration_rate = 0.8

    return True



training_data = json.load(open(config.workspace_dir + "/game_logs_240714_214823.json"))


agent.run(env, 5, train = True)

save_interval = 500

if keep_training(agent):
    agent.run(env, 500, train = True)
else:
    print("Training SUCCEEDED!!!!!")
    # filename = f"tetris_ep{agent.agent_episode_count}_TRAINED.pth"
    # full_path = os.path.join(model_save_dir, filename)
    # torch.save(agent.model.state_dict(), full_path)
    # print(f"Saved model to {full_path}")




# agent.train(env, 10)








