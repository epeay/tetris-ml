from collections import defaultdict, deque
from dataclasses import dataclass
from io import StringIO
import sys
import gymnasium as gym
from pandas import DataFrame
import pandas as pd
import wandb

from tetrisml.base import ActionContext

from .board import TetrisBoard
from .env import ActionFeedback, TetrisEnv
from .tetrominos import Tetrominos
from .minos import MinoShape
from .env import MinoBag
from .base import BaseBag, BaseEnv, ContextPlacement, ModelAction
import random
import numpy as np

random
from tetrisml import board


class DigBag(BaseBag):

    def __init__(self, seed: int = None):
        super().__init__()

    def set_upcoming(self, ms: list[int]) -> None:
        """
        The Dig queue is deterministic and driven by the board configuration.
        ms: A list of mino IDs
        """
        self.clear()
        self.extend(ms)

    def pull(self) -> int:
        return self.popleft()


class DigEnvConfig:
    def __init__(self):
        self.board_height: int = 20
        self.board_width: int = 10
        self.seed = None


class DigEnv(BaseEnv):
    def __init__(self, config: DigEnvConfig):
        super().__init__()
        self.board_height = config.board_height
        self.board_width = config.board_width
        self.seed = config.seed
        self.rng = random.Random(self.seed)

        self.board = DigBoard(
            np.zeros((self.board_height, self.board_width), dtype=int),
            self.board_height,
            seed=self.seed,
        )
        self.bag = DigBag()
        self.stats_table = DataFrame(
            columns=[
                "Shape",
                "Rotation",
                "Column",
                "Correct",
                "Incorrect",
                "Total",
            ]
        )

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self, ctx: ActionContext, title: str = ""):

        m = None
        c = None
        if ctx.placement is not None:
            m = ctx.placement.mino
            c = ctx.placement.coords

        buffer = StringIO()
        sys.stdout = buffer
        TetrisBoard.render_last_action(
            self.board.board,
            m,
            c,
            title=title,
        )
        sys.stdout = sys.__stdout__
        lhs = buffer.getvalue().split("\n")
        max_width = max([len(x) for x in lhs])

        rhs = []

        if ctx.placement is None:
            rhs = [
                f"Current Mino: {Tetrominos.shape_name(self.get_current_mino().shape_id)}",
            ]

        if len(rhs) < len(lhs):
            rhs.extend([""] * (len(lhs) - len(rhs)))

        if len(lhs) < len(rhs):
            lhs.extend([""] * (len(rhs) - len(lhs)))

        for l, r in zip(lhs, rhs):
            print(f"{l:<{max_width}} | {r}")

    def get_debug_dict(self) -> dict:
        return {
            "current_mino": Tetrominos.shape_name(self.get_current_mino().shape_id),
            "reward": self.calculate_reward(),
            "seed": self.board.seed,
        }

    def get_wandb_dict(self) -> dict:
        return self.get_debug_dict()

    def on_before_input(self, ctx: ActionContext, p_ctx: ActionContext):
        self.board.setup()

        # TODO Store board state in context

    def on_action_commit(self, ctx: ActionContext):
        pass

    def on_session_end(self):
        t = wandb.Table(dataframe=self.stats_table)
        wandb.log({"action_stats_table": t})

    def get_current_mino(self) -> MinoShape:
        return self.board.get_current_mino()

    def is_episode_over(self, ctx: ActionContext) -> bool:
        if ctx.valid_action:
            return True  # Default for one-step games

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

    def record_action(self, action: ModelAction, correct: bool):
        table = self.stats_table
        col, rotation = action
        col += 1

        shape_name = Tetrominos.shape_name(self.get_current_mino().shape_id)

        sol: DigBoardSolution = self.board.solution[0]
        outcome = "Correct" if correct else "Incorrect"

        mask = (
            (table["Shape"] == shape_name)
            & (table["Rotation"] == rotation)
            & (table["Column"] == col)
        )

        if not table[mask].empty:
            table.loc[mask, "Total"] += 1
            table.loc[mask, outcome] += 1
        else:
            # I don't think this append happens in place
            new_row = pd.DataFrame(
                [
                    {
                        "Shape": shape_name,
                        "Rotation": rotation,
                        "Column": col,
                        "Correct": 1 if correct else 0,
                        "Incorrect": 1 if not correct else 0,
                        "Total": 1,
                    }
                ]
            )
            self.stats_table = pd.concat([table, new_row], ignore_index=True)

    def commit_action(self, ctx: ActionContext):
        col, rotation = ctx.player_action

        mino = MinoShape(self.get_current_mino().shape_id, rotation)
        lcoords = self.board.find_logical_BL_coords(mino, col)

        ctx.placement = ContextPlacement(mino, lcoords)

        # self.render()

        self.board.place_shape(mino, lcoords)

        correct = self.calculate_reward() == 2.0

        self.record_action(ctx.player_action, correct)

    def post_commit(self, ctx: ActionContext):
        # If any lines are full, save the intermediate state
        line_clear = np.any([sum(x) == self.board_width for x in self.board.board])
        if line_clear:
            ctx.intermediate_board = self.board.export_board()

        ctx.lines_cleared = self.board.remove_tetris()

        # For now, the game only has one board
        ctx.ends_game = True

        # TODO This will need refining
        # if ctx.lines_cleared != 2:
        #     ctx.ends_game = True

    def calculate_reward(self) -> float:
        return board.calculate_reward(self.board.board)


@dataclass
class DigBoardSolution:
    shape: str
    rotation: int
    column: int


class DigBoard(TetrisBoard):
    def __init__(self, matrix: np.ndarray, height: int, seed=None):
        super().__init__(matrix, height)
        self.seed = seed
        self.rng = random.Random(seed)

        # Because this is a puzzle game, the board dictates the mino queue.
        self.mino_queue = deque()

        self.env = None

        self.solution: list[DigBoardSolution] = []

    def __repr__(self) -> str:
        buffer = StringIO()
        sys.stdout = buffer
        self.render()
        sys.stdout = sys.__stdout__
        buffer = buffer.getvalue().split("\n")

        # Brittle code, but it's just for debugging
        # Get last 5 lines

        attrs = ["r1", "r2", "r3", "r4", "r5"]
        buffer = buffer[len(attrs) * -1 :]
        for i, line in enumerate(buffer):
            self.__setattr__(attrs[i], line)

    def do_instrumentation(self, env: TetrisEnv):
        """
        A chance for the board to get access to the environment
        """
        self.env = env
        # env.events.register(E_BEFORE_INPUT, self.setup)
        # env.events.register(E_MINO_SETTLED, env.debug_output)

    def get_current_mino(self) -> MinoShape:
        return self.mino_queue[0]

    def step(self, action):
        pass

    def reset(self, seed=None):
        super().reset()
        if seed is not None:
            self.seed = seed
            self.rng.seed(seed)
        self.setup()

    def setup(self):
        """
        Creates a pattern which the player must dig through to reach the
        bottom of the board, initially using a single move.
        """

        """
        Below are configurations that fit the game format and restrictions of
        the model's actions.

        Example on a Nx10 board
        Given a sideways T
           X
         X X
           X

        The dig board can be set two high.

        X X X X X _ _ X X X
        X X X X X X _ X X X

        And the board can be solved in a single move, if given a T mino (or S,
        for that matter).
        """
        allowed_configurations = [
            (Tetrominos.O, 0),
            (Tetrominos.I, 1),
            (Tetrominos.S, 1),
            (Tetrominos.Z, 1),
            (Tetrominos.J, 1),
            (Tetrominos.L, 2),
            (Tetrominos.T, 1),
            (Tetrominos.T, 3),
        ]

        self.solution = []

        # Pick a winning placement, and construct a board, for that placement.
        chosen_config = self.rng.choice(allowed_configurations)

        # Place the mino on the board, in a random column
        mino = MinoShape(chosen_config[0], chosen_config[1])
        col_range = self.width - mino.width + 1
        col = self.rng.randint(1, col_range)

        shape_name = Tetrominos.shape_name(chosen_config[0])
        self.solution.append(DigBoardSolution(shape_name, chosen_config[1], col))

        stage_board = TetrisBoard(np.zeros((4, self.width), dtype=int), 4)
        stage_board.place_shape(mino, (1, col))

        # Invert board values, 0 -> 1, 1 -> 0
        stage_board.board = np.logical_not(stage_board.board).astype(int)

        # Prep the real board for the user
        self.board.fill(0)

        self.board[0:2] = stage_board.board[0:2]

        if self.env is not None:
            self.env.current_mino = MinoShape(chosen_config[0], 0)

        self.mino_queue.clear()
        self.mino_queue.append(MinoShape(chosen_config[0], 0))
