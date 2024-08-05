import numpy as np
import os
import random
import sys
import time
from datetime import datetime


class GameFrameCollection:
    def __init__(self):
        self.episode_num = 0
        self.player_id = ""
        self.game_id = ""
        self.seeds = []
        self.start_ts = None
        self.frames = []

        self.action_count = 0  # Valid actions
        self.input_count = 0  # Valid and invalid actions

    @property
    def board_height(self) -> int | None:
        return self[0].board.shape[0] if len(self.frames) else None

    @property
    def board_width(self) -> int | None:
        return self[0].board.shape[1] if len(self.frames) else None

    def add_frame(self, frame):
        frame.action_count = self.action_count
        frame.input_count = self.input_count
        self.frames.append(frame)

    def append(self, frame):
        self.add_frame(frame)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

    def __iter__(self):
        return iter(self.frames)


class GameFrame:

    def __init__(self):
        self.board: np.ndarray = None
        self.intermediate_board: np.ndarray = None
        self.mino: str = ""
        self.upcoming_minos = []
        self.action_col = None
        self.action_rot = None

        self.action_count = 0  # How many actions since last action frame
        self.input_count = 0  # How many inputs since last action frame

    def copy_board(self, board: np.ndarray):
        self.board = board.copy()


gfc = GameFrameCollection()
print("done")
