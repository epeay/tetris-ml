import datetime
from json import load
import os
import pprint
import random
import sys

from setup import before_tensorflow  # fmt:skip
before_tensorflow()
import tensorflow as tf  # type: ignore

from setup import make_model_player
import numpy as np
from pandas import DataFrame
import torch
import wandb
from config import Hyperparameters, TMLConfig, load_config
from model import BasePlayer
from model import DQNAgent, ModelPlayer, ModelCheckpoint, TetrisCNN, TetrisCNNConfig
from tetrisml.dig import DigBoard, DigEnv, DigEnvConfig
from tetrisml.env import PlaySession, TetrisEnv
from tetrisml.minos import MinoShape
from tetrisml.tetrominos import Tetrominos

from player import CheatingPlayer, RandomPlayer
from config import load_config, hp
import utils

import time


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)


# set_seed(42)


def run_eval():

    validation_game_seed = "whiff-adorn-1"

    config = load_config()

    hp: Hyperparameters = Hyperparameters()
    hp.game.seed = validation_game_seed

    # Some config values are set at runtime
    hp.model.action_dim = hp.board.width * 4  # 4 mino rotations
    hp.model.linear_data_dim = Tetrominos.get_num_tetrominos()

    wandb.require("core")
    wandb.init(
        project=config.project_name,
        config=hp,
        name=config.run_id,
    )

    mc = TetrisCNNConfig(
        model_id=config.model_id,
        action_dim=hp.model.action_dim,
        dropout_rate=hp.model.dropout_rate,
        board_height=hp.board.height,
        board_width=hp.board.width,
        linear_layer_input_dim=hp.model.linear_data_dim,
    )

    dc = DigEnvConfig(
        board_height=hp.board.height,
        board_width=hp.board.width,
        seed=hp.game.seed,
    )

    ### Config finalized

    model: TetrisCNN = load_model_from_file(
        os.path.join(config.model_storage_dir, "240804-clean-stone.pth"), mc
    )
    p = ModelPlayer(model)
    e = DigEnv(dc)
    sesh = PlaySession(e, p)

    start = config.unix_ts
    sesh.play_game(100)
    print("Done! Elapsed time:", time.time() - start)

    return sesh


# Verify TensorFlow is using CPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


def load_model_from_file(path, config: TetrisCNNConfig) -> TetrisCNN:
    checkpoint: ModelCheckpoint = torch.load(path)
    model = TetrisCNN(config)
    model.load_state_dict(checkpoint["model_state"])
    return model


sesh: PlaySession = run_eval()

stats: DataFrame = sesh.env.stats_table
stats.sort_values(["Shape", "Rotation", "Correct"], inplace=True)

s = stats.groupby(["Shape"]).agg(
    {
        "Total": ["sum"],
    }
)

t = stats.groupby(["Shape", "Rotation", "Column"]).agg(
    {"Total": ["sum"], "Correct": ["sum"], "Incorrect": ["sum"]}
)
t["pct"] = t["Correct"] / t["Total"]

print(t.to_string())
print("done")
#########################################
