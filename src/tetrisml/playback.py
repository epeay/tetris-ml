import numpy as np
import os
import random
import sys
import time
from datetime import datetime


class GameFrameCollection:
    def __init__(self):
        self.seeds = []
        self.start = None
        self.frames = []

    @property
    def board_height(self) -> int | None:
        return self[0].board.shape[0] if len(self.frames) else None

    @property
    def board_width(self) -> int | None:
        return self[0].board.shape[1] if len(self.frames) else None

    def add_frame(self, frame):
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

    def __init__(self, board, mino: str):
        self.board: np.ndarray = board
        self.mino: str = mino
        self.upcoming_minos = []
        self.mino_rot = mino_rot
        self.input_col = None
        self.input_rot = None

    def store_board(self, board: np.ndarray):
        self.board = board.copy()


gfc = GameFrameCollection()
print("done")
