from ast import Str
from collections import deque
from io import StringIO
import sys
import gymnasium as gym

from tetrisml.base import ActionContext

from .board import TetrisBoard
from .env import E_BEFORE_INPUT, E_MINO_SETTLED, ActionFeedback, TetrisEnv
from .tetrominos import Tetrominos
from .minos import MinoShape
from .env import MinoBag
from .base import BaseBag, BaseEnv, ContextPlacement
import random
import numpy as np

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


class DigEnv(BaseEnv):
    def __init__(self):
        self.board_height = 10
        self.board_width = 5
        self.board = DigBoard(
            np.zeros((self.board_height, self.board_width), dtype=int),
            self.board_height,
        )
        self.bag = DigBag()

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

    def debug_output(self):
        lhs = StringIO()
        rhs = StringIO()

        sys.stdout = lhs
        TetrisBoard.render_last_action(
            self.board.board, placed_mino, lcoords, title="Debug Output"
        )
        sys.stdout = sys.__stdout__
        lhs = lhs.getvalue().split("\n")
        max_lhs = max([len(x) for x in lhs])

        sys.stdout = rhs
        print(f"Current Mino: {self.current_mino}")
        sys.stdout = sys.__stdout__
        rhs = rhs.getvalue().split("\n")

        for l, r in zip(lhs, rhs):
            print(f"{l:<{max_lhs}} | {r}")

    def on_before_input(self, ctx: ActionContext, p_ctx: ActionContext):
        self.board.setup()

        # TODO Store board state in context

    def get_current_mino(self) -> MinoShape:
        return self.board.get_current_mino()

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
            ctx.valid_action = False
            return False, {
                "message": f"Piece {self.current_mino} at {lcol} overflows the board"
            }

        ctx.valid_action = True

    def commit_action(self, ctx: ActionContext):
        col, rotation = ctx.player_action
        done = False

        mino = MinoShape(self.get_current_mino().shape_id, rotation)
        lcoords = self.board.find_logical_BL_coords(mino, col)

        ctx.placement = ContextPlacement(mino, lcoords)

        # self.render()

        self.board.place_shape(mino, lcoords)
        # self.events.call(E_MINO_SETTLED, mino, lcoords)

        # If any of the top four rows were used -- Game Over
        # TODO Evaluate for a board that doesn't have the extra rows
        if np.any(self.board.board[-4:]):
            # Game Over
            done = True
            reward = -1

            # self.record.rewards.append(reward)
            # self.record.cumulative_reward += reward
            # self.reward_history.append(reward)

            print("CLOSING EPISODE")

            self.close_episode()

            return done

        # reward = self.calculate_reward()
        done = False

        # self.record.rewards.append(reward)
        # self.record.cumulative_reward += reward
        # self.reward_history.append(reward)

    def post_commit(self, ctx: ActionContext):
        # If any lines are full, save the intermediate state
        line_clear = np.any([sum(x) == self.board_width for x in self.board.board])
        if line_clear:
            ctx.intermediate_board = self.board.export_board()

        ctx.lines_cleared = self.board.remove_tetris()

        # TODO This will need refining
        if ctx.lines_cleared != 2:
            ctx.ends_game = True

    def calculate_reward(self) -> float:
        return board.calculate_reward(self.board.board)


class DigBoard(TetrisBoard):
    def __init__(self, matrix: np.ndarray, height: int, seed: int = None):
        super().__init__(matrix, height)
        self.seed = seed
        self.rng = random.Random(seed)

        # Because this is a puzzle game, the board dictates the mino queue.
        self.mino_queue = deque()

        self.env = None

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

    def reset(self, seed: int = None):
        super().reset()
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
            # (Tetrominos.L, 2),
            # (Tetrominos.T, 1),
            # (Tetrominos.T, 3),
        ]

        # Pick a winning placement, and construct a board, for that placement.
        chosen_config = self.rng.choice(allowed_configurations)

        # Place the mino on the board, in a random column
        mino = MinoShape(chosen_config[0], chosen_config[1])
        col_range = self.width - mino.width + 1
        col = random.randint(1, col_range)
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
