import datetime
import gymnasium as gym
import numpy as np
import time
import random
import pytest

from collections import deque
from datetime import datetime
from gymnasium import spaces

from tetrisml.tetrominos import Tetrominos, TetrominoPiece
from tetrisml.minos import MinoPlacement, MinoShape
from tetrisml.board import TetrisBoard
from tetrisml.logging import TetrisGameRecord

import numpy as np
from numpy import ndarray

from typing import NewType

ModelAction = NewType("ModelAction", tuple[int, int])


class ActionFeedback:
    def __init__(self, valid_action=False):
        # Does this action place the mino on or above the board, and not
        # embedded in a wall, for example.
        self.valid_action = valid_action
        self.is_predict = False
        self.lines_cleared = 0
        # If lines_cleared, this will show the board state before the lines
        # were cleared. Useful for model training.
        self.intermediate_board: ndarray = None

    def __str__(self):
        return f"ActionFeedback(valid_action={self.valid_action}, is_predict={self.is_predict})"


class EnvStats:
    """ """

    def __init__(self):
        # Use a parlance that isn't tied to an ML model
        self.total_placements = 0
        self.total_games_completed = 0
        self.total_lines_cleared = 0


E_STEP_COMPLETE = "step_complete"  # Steps include invalid moves
E_MINO_SETTLED = "mino_settled"
E_GAME_OVER = "game_over"
E_GAME_START = "game_start"
E_LINE_CLEAR = "line_clear"


class CallbackHandler:
    """
    Allows external entities to register callbacks for various events.
    """

    def __init__(self):
        self.callbacks = {}

    def register(self, event: str, callback):
        if event not in self.callbacks:
            self.callbacks[event] = []

        self.callbacks[event].append(callback)

    def call(self, event: str, *args, **kwargs):
        if event not in self.callbacks:
            return

        for cb in self.callbacks[event]:
            cb(*args, **kwargs)


class TetrisEnv(gym.Env):

    def __init__(self, piece_bag=None, board_height=20, board_width=10):
        """
        board_height: The DESIRED height of the board. The actual height will be
                        board_height + 4 to make clearance for overflows.
        """
        super(TetrisEnv, self).__init__()
        self.board_height = board_height
        self.board_width = board_width
        self.current_mino: MinoShape = None
        self.pieces = Tetrominos()
        self.reward_history = deque(maxlen=10)
        self.record = TetrisGameRecord()
        self.piece_bag = Tetrominos.std_bag if piece_bag is None else piece_bag
        self.step_history: list[MinoPlacement] = []
        self.random: random.Random = None
        self.random_seed = None
        self.stats = EnvStats()
        self.record: TetrisGameRecord = None

        self.events = CallbackHandler()

        # Indexes  0-19 - The visible playfield
        #         20-23 - Buffer for overflows
        self.state = np.zeros((self.board_height + 4, self.board_width), dtype=int)

        # Creates a *view* from the larger state
        # of all but the top four rows
        self.current_piece_rows = self.state[:-4]
        self.board = TetrisBoard(self.state, self.board_height - 4)

        # Action space: tuple (column, rotation)
        self.action_space = spaces.MultiDiscrete([self.board_width, 4])

        self.reset()

    def reset(self, seed: int = None):
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

        self.current_mino = self._get_random_piece()
        self.record = TetrisGameRecord()
        self.step_history: list[MinoPlacement] = []

        return self._get_board_state()

    def is_valid_action(self, action: ModelAction) -> tuple[bool, dict]:
        """
        action: tuple[int, int]
        """
        col, rotation = action
        lcol = col + 1

        # ([0-9], [0-3])
        if lcol < 1 or lcol > self.board_width:
            return False, {"message": "Column out of bounds"}

        # An O piece on col 1 would occupy cols 1-2
        if lcol + self.current_mino.width - 1 > self.board_width:
            return False, {"message": "Piece overflows the board"}

        return True, {}

    def commit_action(self, action: ModelAction) -> tuple[bool, dict]:
        """
        action: tuple[int, int]
        """
        col, rotation = action
        done = False

        info = ActionFeedback()
        mino = MinoShape(self.current_mino.shape_id, rotation)

        info.valid_action = True
        lcoords = None

        lcoords = self.board.find_logical_BL_coords(mino, col)
        self.board.place_shape(mino, lcoords)

        self.events.call(E_MINO_SETTLED)

        # self.stats.total_placements += 1
        # self.record.moves += 1
        # self.record.boards.append(self.board.board.copy())
        # self.record.pieces.append(mino.get_piece().to_dict())
        # self.current_mino.rot = 0
        # self.record.placements.append(lcoords)

        # If any of the top four rows were used -- Game Over
        if np.any(self.board.board[-4:]):
            # Game Over
            done = True
            reward = -1

            self.record.rewards.append(reward)
            self.record.cumulative_reward += reward
            self.reward_history.append(reward)

            self.close_episode()

            return done, info

        # reward = self.calculate_reward()
        done = False

        # self.record.rewards.append(reward)
        # self.record.cumulative_reward += reward
        # self.reward_history.append(reward)

        # If any lines are full
        self.board.remove_tetris()

        return done, info

    def step(self, action: ModelAction):
        """
        action: zero-indexed tuple of (column, rotation)
        """
        # ([0-9], [0-3])
        col, rotation = action
        lcol = col + 1

        info = ActionFeedback()
        mino = MinoShape(self.current_mino.shape_id, rotation)

        # Clear the area above the visible board. If this range is used during
        # piece placement, the game is over.
        self.board.board[-4:].fill(0)

        # Check for right-side overflow
        # Given a horizontal I piece on col 0
        # right_lcol would be 4. The piece would occupy lcolumns 1-4.
        right_lcol = col + mino.width
        if right_lcol > self.board_width:
            # Ignore this action and try again.
            #
            # For example, a location is chosen which extends
            # the piece over the edge of the board.
            done = False
            info.valid_action = False
            self.record.invalid_moves += 1
            reward = -1

            return self._get_board_state(), reward, done, info

        info.valid_action = True
        lcoords = None

        lcoords = self.board.find_logical_BL_coords(mino, col)
        self.board.place_shape(mino, lcoords)

        self.events.call(E_MINO_SETTLED)

        self.stats.total_placements += 1
        self.record.moves += 1
        self.record.boards.append(self.board.board.copy())
        self.record.pieces.append(mino.get_piece().to_dict())
        self.current_mino.rot = 0

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

        reward = self.calculate_reward()
        done = False

        if reward < 1:
            reward = -1

        done = True

        self.record.rewards.append(reward)
        self.record.cumulative_reward += reward
        self.reward_history.append(reward)

        # If any lines are full
        line_clear = np.any([sum(x) == self.board_width for x in self.board.board])

        if line_clear:
            # TODO Send the board before and after the clears
            self.events.call(E_LINE_CLEAR)

        # Huzzah!
        lines_gone = self.board.remove_tetris()
        if lines_gone > 0:
            self.record.cleared_by_size[lines_gone] += 1

        self.stats.total_lines_cleared += lines_gone
        self.record.lines_cleared += lines_gone

        # Prep for next move
        self.current_mino = self._get_random_piece()
        next_board_state = self._get_board_state()
        return next_board_state, reward, True, info

    def close_episode(self):
        """
        Wraps up episode stats. Public method is available for agent to call
        if needed.
        """

        if self.record.episode_end_time:
            # Already closed
            return

        self.record.episode_end_time = time.monotonic_ns()
        self.record.duration_ns = (
            self.record.episode_end_time - self.record.episode_start_time
        )
        self.stats.total_games_completed += 1

    def render(self):
        self.board.render()

    def _get_random_piece(self):
        return MinoShape(self.random.choice(self.piece_bag))

    def _is_valid_action(self, piece, lcol):
        piece = self.current_mino

        if lcol < 1 or lcol > self.board_width:
            return False

        # An O piece on col 1 would occupy cols 1-2
        if lcol + piece.get_width() - 1 > self.board_width:
            return False
        return True

    def calculate_reward(self):
        tower_height = 0

        line_pack = []
        clears = 0

        for r in self.board.board:
            pack = sum(r)
            if pack == 0:
                break

            if pack == self.board_width:
                clears += 1

            line_pack.append(sum(r))
            tower_height += 1

        pct_board_full = sum(line_pack) / (self.board_width * tower_height)
        return max(clears, pct_board_full)

    def _get_board_state(self) -> np.ndarray:
        """
        Returns a copy of the board state. Shape is (1, H, W)
        """
        # Copy the board state
        # INEFFICIENT!!!
        # TODO Do I still need to do this?
        state = self.board.board.copy()
        return state[np.newaxis, :, :]

    @staticmethod
    def smoltris():
        bag = [Tetrominos.O, Tetrominos.DOT, Tetrominos.USCORE]
        return TetrisEnv(piece_bag=bag, board_height=10, board_width=5)

    @staticmethod
    def tetris():
        return TetrisEnv()


class MinoBag(deque):
    def __init__(self, tiles: list[int], seed: int, maxlen: int = 10):
        super().__init__(maxlen=maxlen)
        self.tiles = tiles
        self.seed: int = None
        self.r = random.Random(seed)
        self.populate()

    def populate(self):
        while len(self) < self.maxlen:
            self.append(self.r.choice(self.tiles))

    def popleft(self):
        ret = super().popleft()
        self.populate()
        return ret

    def pull(self):
        return self.popleft()

    def __str__(self):
        return f"MinoBag({[Tetrominos.shape_name(x) for x in self]})"


class BasePlayer:
    def __init__(self):
        pass

    def play(self, e: TetrisEnv) -> ModelAction:
        pass

    def on_episode_start(self):
        pass

    def on_episode_end(self):
        """Includes reason for termination"""
        pass


# play.sess.shon.
class PlaySession:
    """
    A simple connector between player and environment
    """

    def __init__(self, e: TetrisEnv, p: BasePlayer):
        self.env = e
        self.player: BasePlayer = p
        self.events = CallbackHandler()

    def play_game(self, episodes: int = 1):
        self.env.reset()

        while True:
            action = self.player.play(self.env)

            # Can this move be accepted by the environment?
            valid, info = self.env.is_valid_action(action)

            if not valid:
                raise ValueError(f"Invalid action: {info}")

            done, info = self.env.commit_action(action)
            self.player.on_action_commit(self.env, action, done)

            if done:
                break
