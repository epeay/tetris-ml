from ast import Call
from collections import deque
from typing import NewType

from numpy import ndarray

from .minos import MinoShape

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

        # A pocket dimension for the env to store anything else it needs.
        self.env_ctx: object = None
        # A pocket dimension for the player to store anything else it needs.
        self.player_ctx: object = None


class BaseBoard:
    def __init__(self, matrix: ndarray, height):
        self.play_height = height
        self.height = len(matrix)
        self.width = len(matrix[0])
        self.board: ndarray = matrix

    def export_board(self):
        raise NotImplementedError("export_board() must be implemented by subclass")


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


class BaseEnv:
    def __init__(self):
        self.board_height: int = 0
        self.board_width: int = 0
        self.board: BaseBoard = None
        self.events: CallbackHandler = CallbackHandler()

    def get_current_mino(self) -> MinoShape:
        raise NotImplementedError("get_current_mino() must be implemented by subclass")

    def reset(self, *args, **kwargs):
        raise NotImplementedError("reset() must be implemented by subclass")

    def step(self, *args, **kwargs):
        raise NotImplementedError("step() must be implemented by subclass")

    def render(self, *args, **kwargs):
        raise NotImplementedError("render() must be implemented by subclass")

    def debug_output(self, *args, **kwargs):
        raise NotImplementedError("debug_output() must be implemented by subclass")

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


class BasePlayer:
    def __init__(self):
        pass

    def play(self, e: BaseEnv) -> ModelAction:
        pass

    def on_episode_start(self, e: BaseEnv):
        pass

    def on_episode_end(self, e: BaseEnv):
        pass

    def on_invalid_input(self, ctx: ActionContext):
        pass

    def on_action_commit(self, e: BaseEnv, action: ModelAction, done: bool):
        pass

    def get_debug_dict(self):
        return {}
