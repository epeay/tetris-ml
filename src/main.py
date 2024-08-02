import datetime
import json
import numpy as np
import os
import sys

import matplotlib.pyplot as plt

from config import load_config, TMLConfig
from tetrisml import *
from model import *
from tetrisml.gamesets import GameRuns
import utils
from cheating import *
from viz import *
import time
from tetrisml.env import PlaySession, TetrisEnv
from tetrisml import MinoShape, Tetrominos, GameHistory
from player import CheatingPlayer, PlaybackPlayer
from model import DQNAgent

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf  # type: ignore

# Verify TensorFlow is using CPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

"""
Episode = One tetris game
mino = Tetromino
"""

config: TMLConfig = load_config()
game_logs: list[GameHistory] = []


def save_game_logs(game_logs: list[GameHistory], path: str = "game_logs.json"):
    ret = {"games": {}}

    for game in game_logs:
        ret["games"][game.id] = game.to_jsonable()

    print(f"Saving game logs to {path}")
    with open(path, "w") as outfile:
        json.dump(ret, outfile, indent=4)
