from argparse import Action
import datetime
from io import StringIO
import sys
from telnetlib import GA
from flask import g
import gymnasium as gym
import numpy as np
import time
import random
import pytest

from collections import deque
from datetime import datetime
from gymnasium import spaces
import wandb

from tetrisml.base import BaseBag, BaseEnv, ActionContext, BasePlayer, ModelAction
from tetrisml.display import GridRenderConfig, GridRenderer
from tetrisml.tetrominos import Tetrominos, TetrominoPiece
from tetrisml.minos import MinoPlacement, MinoShape
from tetrisml.board import TetrisBoard
from tetrisml.tetris_logging import TetrisGameRecord
from tetrisml.playback import GameFrameCollection, GameFrame

from .base import CallbackHandler, EpisodeContext

import numpy as np
from numpy import ndarray

from typing import NewType


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


class TetrisEnv(BaseEnv):

    E_BEFORE_INPUT = "before_input"
    E_ACTION_COMMITED = "action_committed"

    def __init__(self, piece_selection, board_height=20, board_width=10):
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
        # TODO handle seed
        self.piece_bag: BaseBag = MinoBag(piece_selection, None)
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

    def get_current_mino(self) -> MinoShape:
        return self.current_mino

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

    def is_valid_action(self, ctx: ActionContext) -> tuple[bool, dict]:
        """
        action: tuple[int, int]
        """
        col, rotation = ctx.player_action
        lcol = col + 1

        # ([0-9], [0-3])
        if lcol < 1 or lcol > self.board_width:
            return False, {"message": "Column out of bounds"}

        pending_mino = MinoShape(self.current_mino.shape_id, rotation)

        # An O piece on col 1 would occupy cols 1-2
        if lcol + pending_mino.width - 1 > self.board_width:
            return False, {
                "message": f"Piece {self.current_mino} at {lcol} overflows the board"
            }

        return True, {}

    def on_before_input(self, ctx: ActionContext, p_ctx: ActionContext):
        # self.events.call(E_BEFORE_INPUT)
        pass

    def on_action_commit(self, ctx: ActionContext):
        pass

    def debug_output(self, placed_mino, lcoords):
        pass

    def get_debug_dict(self) -> dict:
        return {}

    def commit_action(self, ctx: ActionContext) -> tuple[bool, dict]:
        """
        action: tuple[int, int]
        """
        col, rotation = ctx.player_action
        done = False

        info = ActionFeedback()
        mino = MinoShape(self.current_mino.shape_id, rotation)

        info.valid_action = True
        lcoords = None

        lcoords = self.board.find_logical_BL_coords(mino, col)

        self.render()

        self.board.place_shape(mino, lcoords)
        # self.events.call(E_MINO_SETTLED, mino, lcoords)

        # self.stats.total_placements += 1
        # self.record.moves += 1
        # self.record.boards.append(self.board.board.copy())
        # self.record.pieces.append(mino.get_piece().to_dict())
        # self.current_mino.rot = 0
        # self.record.placements.append(lcoords)

        # If any of the top four rows were used -- Game Over
        if np.any(self.board.board[-4:]):
            # Game Over
            ctx.ends_game = True
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

        # If any lines are full, save the intermediate state
        line_clear = np.any([sum(x) == self.board_width for x in self.board.board])
        if line_clear:
            info.intermediate_board = self.board.export_board()

        info.lines_cleared = self.board.remove_tetris()

        return done, info

    def post_commit(self, ctx: ActionContext):
        self.current_mino = self._get_random_piece()

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
        return MinoShape(self.piece_bag.pull())

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
        return TetrisEnv(piece_selection=bag, board_height=10, board_width=5)

    @staticmethod
    def tetris():
        return TetrisEnv(Tetrominos.std_bag)


class MinoBag(deque):
    def __init__(self, tiles: list[int], seed: int, maxlen: int = 10):
        super().__init__(maxlen=maxlen)
        self.tiles = tiles
        self.seed: int = None
        self.rng = random.Random(seed)
        self.populate()

    def populate(self):
        while len(self) < self.maxlen:
            self.append(self.rng.choice(self.tiles))

    def popleft(self):
        ret = super().popleft()
        self.populate()
        return ret

    def pull(self):
        return self.popleft()

    def __str__(self):
        return f"MinoBag({[Tetrominos.shape_name(x) for x in self]})"


# play.sess.shon.
class PlaySession:
    """
    A simple connector between player and environment
    """

    E_ACTION_COMMITTED = "action_committed"

    def __init__(self, e: BaseEnv, p: BasePlayer):
        self.env: BaseEnv = e
        self.player: BasePlayer = p
        self.events = CallbackHandler()

        self.episode_num: int = 0
        self.committed_action_num: int = 0

        self.session_games: list[GameFrameCollection] = []

    def _render_dict(self, indent: str, d: dict):
        ret = []
        for k, v in d.items():
            if isinstance(v, dict):
                ret.append(f"{indent}{k}:")
                ret += self._render_dict(indent + "  ", v)
            else:
                ret.append(f"{indent}{k}: {v}")

        return ret

    def render(self, last_placement: MinoPlacement = None):
        """
        Produces a two column output. The left column is the board state. The
        right column is a debug output from the player and environment.

        Example:
        0 _ _ _ _ _  | Player Info:
        9 _ _ _ _ _  |   exploration_rate: 1.0
        8 _ _ _ _ _  |   replay_buffer_size: 120
        7 _ _ _ _ _  |   replayed steps: 0
        6 ▆ _ _ _ _  | Env Info:
        5 ▆ _ _ _ _  |   current_mino: I
        4 ▆ _ _ _ _  |
        3 ▆ _ _ _ _  |
        2 ▆ ▆ ▆ _ ▆  |
        1 ▆ ▆ ▆ _ ▆  |
                     |
        """
        player = self.player.get_debug_dict()
        env = self.env.get_debug_dict()
        gr = GridRenderer(GridRenderConfig())

        ms = None
        lcoords = None
        if last_placement:
            ms = last_placement.mino
            lcoords = last_placement.coords

        buffer = StringIO()
        sys.stdout = buffer

        gr.render(self.env.board.board, ms, lcoords)

        # self.env.board.render()
        sys.stdout = sys.__stdout__
        lhs = buffer.getvalue().split("\n")
        max_width = max([len(x) for x in lhs])

        header = f"Ep {self.episode_num} | Ac {self.committed_action_num}"

        rhs = []
        rhs.append(header)

        rhs.append("Player Info:")
        rhs += self._render_dict("", player)

        # for k, v in player.items():
        #     rhs.append(f"  {k}: {v}")

        rhs.append("Env Info:")
        for k, v in env.items():
            rhs.append(f"  {k}: {v}")

        lhs.append("")
        rhs.append("")

        if len(rhs) < len(lhs):
            rhs.extend([""] * (len(lhs) - len(rhs)))

        if len(lhs) < len(rhs):
            lhs.extend([""] * (len(rhs) - len(lhs)))

        for l, r in zip(lhs, rhs):
            print(f"{l:<{max_width}} | {r}")

    def play_game(self, episodes: int = 1, render: bool = True):
        self.env.board.do_instrumentation(self.env)

        for e in range(episodes):
            self.episode_num += 1

            e_ctx = EpisodeContext()

            e_ctx.game_frames = GameFrameCollection()
            gfc = e_ctx.game_frames
            gfc.episode_num = self.episode_num

            self.committed_action_num = 0
            self.env.reset()

            frame = GameFrame()
            frame.copy_board(self.env.board.board)
            gfc.add_frame(frame)

            last_context = None
            self.player.on_episode_start(self.env)

            while True:
                ctx = ActionContext()
                ctx.session = self
                frame = GameFrame()

                self.env.on_before_input(ctx, last_context)
                del last_context

                frame.mino_queue = self.env.mino_queue

                while ctx.valid_action is not True:
                    ctx.player_action = self.player.play(self.env)
                    gfc.input_count += 1
                    ctx.valid_action, info = self.env.is_valid_action(ctx)
                    # ctx.ends_game = self.env.is_episode_over(ctx)

                    if not ctx.valid_action:
                        # self.env.recover_from_invalid_action(ctx, info)
                        self.player.on_invalid_input(ctx)

                gfc.action_count += 1
                frame.action_col, frame.action_rot = ctx.player_action

                ###
                # Ready to commit user input
                ###

                self.env.commit_action(ctx)
                self.committed_action_num += 1

                frame.board = self.env.board.export_board()

                self.player.on_action_commit(self.env, ctx, ctx.player_ctx)
                self.env.on_action_commit(ctx)
                self.events.call(PlaySession.E_ACTION_COMMITTED, self, ctx)

                p_dict = self.player.get_wandb_dict()
                e_dict = self.env.get_wandb_dict()

                wandb.log(
                    {
                        "episode": self.episode_num,
                        "action": self.committed_action_num,
                        "reward": self.env.calculate_reward(),
                        "player": p_dict,
                        "env": e_dict,
                    }
                )

                if render:
                    self.render(last_placement=ctx.placement)

                full_rows = self.env.board.count_full_rows()
                e_ctx.lines_cleared += full_rows
                frame.lines_cleared = full_rows

                frame.board = self.env.board.export_board()
                if full_rows:
                    frame.intermediate_board = frame.board
                    frame.board = self.env.board.export_board()

                self.env.post_commit(ctx)
                # self.player.on_post_commit(ctx)

                gfc.append(frame)

                # After board clean-up, clear the placement data so that it isn't
                # used to highlight board placements on rows that have been cleared.
                # This feels like the wrong solution.
                ctx.placement = None

                last_context = ctx

                if last_context.ends_game:
                    e_ctx.game_over = True
                    break

            if not render:
                if e_ctx.lines_cleared == 2:
                    print("!", end="")
                else:
                    print(".", end="")

            # Game Over
            self.env.on_episode_end(ctx, e_ctx)
            self.player.on_episode_end(ctx, e_ctx, self.env)
            self.session_games.append(gfc)

        self.env.on_session_end()
