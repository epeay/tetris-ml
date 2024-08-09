from ast import Call
from collections import deque
from dataclasses import dataclass, field
from typing import NewType
import gymnasium as gym

from numpy import ndarray
import numpy as np

from tetrisml.minos import MinoShape
from tetrisml.playback import GameFrameCollection

ModelAction = NewType("ModelAction", tuple[int, int])


class ContextPlacement:
    def __init__(self, mino: MinoShape, coords: tuple[int, int]):
        self.mino: int = mino
        self.coords: tuple[int, int] = coords


class BaseBag(deque):
    def __init__(self):
        super().__init__()

    def pull(self) -> int:
        raise NotImplementedError("pull() must be implemented by subclass")

    def peek(length: int = None) -> int:
        raise NotImplementedError(f"peek() must be implemented by subclass")


@dataclass
class EpisodeContext:
    accurate: bool = True
    game_over: bool = False
    lines_cleared: int = 0
    game_frames: GameFrameCollection = field(default_factory=GameFrameCollection)


class ActionContext:

    def __init__(self):
        self.starting_board = None
        self.player_action: ModelAction = None
        self.valid_action: bool = None
        self.placement: ContextPlacement = None
        self.lines_cleared: int = None
        # Used to store the board before clearing any lines, otherwise None
        self.intermediate_board = None
        self.final_board = None
        self.ends_game: bool = False
        self.session = None  # Temporary Hack

        # A pocket dimension for the env to store anything else it needs.
        self.env_ctx: object = None
        # A pocket dimension for the player to store anything else it needs.
        self.player_ctx: object = None


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


class BaseEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board_height: int = 0
        self.board_width: int = 0
        self.board: BaseBoard = None
        self.events: CallbackHandler = CallbackHandler()

    def get_current_mino(self) -> MinoShape:
        raise NotImplementedError("get_current_mino() must be implemented by subclass")

    @property
    def mino_queue(self) -> deque[int]:
        raise NotImplementedError("mino_queue() must be implemented by subclass")

    def reset(self, *args, **kwargs):
        raise NotImplementedError("reset() must be implemented by subclass")

    def step(self, *args, **kwargs):
        raise NotImplementedError("step() must be implemented by subclass")

    def render(self, *args, **kwargs):
        raise NotImplementedError("render() must be implemented by subclass")

    def debug_output(self, *args, **kwargs):
        raise NotImplementedError("debug_output() must be implemented by subclass")

    def on_action_commit(self, ctx: ActionContext):
        raise NotImplementedError("on_action_commit() must be implemented by subclass")

    def on_before_input(self, ctx: ActionContext, p_ctx: ActionContext):
        raise NotImplementedError("on_before_input() must be implemented by subclass")

    def is_episode_over(self, ctx: ActionContext):
        raise NotImplementedError("is_episode_over() must be implemented by subclass")

    def is_valid_action(self, ctx: ActionContext):
        raise NotImplementedError("is_valid_action() must be implemented by subclass")

    def commit_action(self, ctx: ActionContext):
        raise NotImplementedError("commit_action() must be implemented by subclass")

    def calculate_reward(self) -> float:
        raise NotImplementedError("calculate_reward() must be implemented by subclass")

    def get_debug_dict(self) -> dict:
        raise NotImplementedError("get_debug_dict() must be implemented by subclass")

    def get_wandb_dict(self):
        return {}

    def post_commit(self, ctx: ActionContext):
        pass

    def on_session_end(self):
        pass


class BaseBoard:
    def __init__(self, matrix: ndarray, height):
        self.play_height = height

    @property
    def height(self):
        return self.board.shape[0]

    @property
    def rows(self):
        return self.height

    @property
    def width(self):
        return self.board.shape[1]

    @property
    def cols(self):
        return self.width

    def export_board(self):
        return BaseBoard(self.board.copy())

    def get_pack(self) -> ndarray:
        """
        Pack refers to the number of filled cells in a given row. Returns the
        pack of each row.
        """
        return self.board.sum(1)

    def count_full_rows(self):
        pack = self.get_pack()
        return len([x for x in pack if x == self.width])

    def do_instrumentation(self, env: BaseEnv):
        pass


class BasePlayer:
    def __init__(self):
        pass

    def play(self, e: BaseEnv) -> ModelAction:
        pass

    def on_episode_start(self, e: BaseEnv):
        pass

    def on_episode_end(self, ctx: ActionContext, e_ctx: EpisodeContext, e: BaseEnv):
        pass

    def on_invalid_input(self, ctx: ActionContext):
        pass

    def on_action_commit(self, e: BaseEnv, action: ModelAction, done: bool):
        pass

    def get_debug_dict(self):
        return {}

    def get_wandb_dict(self):
        return {}
