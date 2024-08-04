import datetime
import os
import pprint
import random
import sys
import time

import wandb
from model import DQNAgent, DQNAgentConfig, TetrisCNN, TetrisCNNConfig
from tetrisml.dig import DigBoard, DigEnv, DigEnvConfig
from tetrisml.env import PlaySession, TetrisEnv
from tetrisml.minos import MinoShape
from tetrisml.tetrominos import Tetrominos

from player import CheatingPlayer, RandomPlayer
from config import Hyperparameters, TMLConfig, load_config, hp
import utils

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Must come after setting these vars
import tensorflow as tf  # type: ignore

assert len(tf.config.experimental.list_physical_devices("GPU")) == 0


sweep_config = {
    "name": "dig-sweep",
    "method": "random",
    "metric": {"name": "player.loss", "goal": "minimize"},
    "parameters": {
        "lr": {"max": 0.1, "min": 0.001},
        "dropout": {"max": 0.5, "min": 0.0},
    },
}


wandb.require("core")
p = None


def make_model_player(cfg: TMLConfig, hp: Hyperparameters) -> DQNAgent:
    # Keeping this import separate because it's not always needed
    # and tensorflow is slow to import
    from model import DQNAgent

    action_dim = hp.board.width * 4  # 4 mino rotations

    # fmt: off
    ac = DQNAgentConfig(
        input_channels     = hp.model.input_channels,
        board_height       = hp.board.height,
        board_width        = hp.board.width,
        action_dim         = hp.model.action_dim,
        linear_data_dim    = hp.model.linear_data_dim,
        model_id           = cfg.model_id,
        exploration_rate   = hp.agent.exploration_rate,
        exploration_decay  = hp.agent.exploration_decay,
        learning_rate      = hp.agent.learning_rate,
        batch_size         = hp.agent.batch_size,
        replay_buffer_size = hp.agent.replay_buffer_size
    )

    mc = TetrisCNNConfig(
        model_id        = cfg.model_id,
        action_dim      = action_dim,
        dropout_rate    = hp.model.dropout_rate,
        board_height    = hp.board.height,
        board_width     = hp.board.width,
        linear_layer_input_dim = hp.model.linear_data_dim,
    )
    # fmt: on

    model = TetrisCNN(mc)
    target_model = TetrisCNN(mc)
    p = DQNAgent(ac, model, target_model)

    return p


def run_sweep():

    config: TMLConfig = load_config()  # Generates a new run_id
    hp: Hyperparameters = Hyperparameters()
    hp.game.seed = config.model_id

    # Some config values are set at runtime
    hp.model.action_dim = hp.board.width * 4  # 4 mino rotations
    hp.model.linear_data_dim = Tetrominos.get_num_tetrominos()

    wandb_config = {
        "cfg": config,
        "hp": hp,
    }
    wandb.init(
        project=config.project_name,
        config=wandb_config,
        # name=config.run_id,
    )

    # Apply sweep overrides
    hp.agent.learning_rate = wandb.config.lr
    hp.model.dropout_rate = wandb.config.dropout

    wandb.config.update(wandb_config, allow_val_change=True)

    e = DigEnv(
        DigEnvConfig(
            board_height=hp.board.height,
            board_width=hp.board.width,
            seed=hp.game.seed,
        )
    )

    p = make_model_player(config, hp)
    sesh = PlaySession(e, p)

    start = config.unix_ts

    while p.exploration_rate > 0.1:
        sesh.play_game(100, render=False)

    print("Trained! Elapsed time:", time.time() - start)


sweep_id = wandb.sweep(sweep_config, project="tetris-ml")

wandb.agent(sweep_id, function=run_sweep, count=30)
