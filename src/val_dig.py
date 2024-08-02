import datetime
import os
import pprint
import random
import sys

import wandb
from model import DQNAgent, ModelPlayer
from tetrisml.dig import DigBoard, DigEnv, DigEnvConfig
from tetrisml.env import PlaySession, TetrisEnv
from tetrisml.minos import MinoShape
from tetrisml.tetrominos import Tetrominos

from player import CheatingPlayer, RandomPlayer
from config import load_config, hp
import utils

import time

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf  # type: ignore

# Verify TensorFlow is using CPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


wandb.require("core")

config = load_config()

hyper = hp.copy()
hyper["env"]["git_hash_short"] = config.git_short
hyper["env"]["unix_ts"] = datetime.datetime.now().timestamp()


model_id = utils.word_id()
hyper["game"]["seed"] = model_id


wandb.init(
    project=hyper["env"]["project_name"],
    config=hyper,
)

h = hyper["board"]["height"] = 20
w = hyper["board"]["width"] = 10


p = None
input_channels = 1
action_dim = 4 * w  # 4 rotations
linear_data_dim = Tetrominos.get_num_tetrominos()


dc = DigEnvConfig()
dc.board_height = h
dc.board_width = w
dc.seed = model_id


e = DigEnv(dc)


def make_model_player():
    # Keeping this import separate because it's not always needed
    # and tensorflow is slow to import
    from model import DQNAgent

    p = DQNAgent(
        input_channels,
        h,
        w,
        action_dim,
        linear_data_dim=linear_data_dim,
        model_id=model_id,
    )

    return p


# p = RandomPlayer()

p = make_model_player()
p.load_model(os.path.join(config.model_storage_dir, "240802-stiff-field.pth"))
p = ModelPlayer(p.model)

p.model.eval()

validation_game_seed = "whiff-adorn-1"  # hardcoded for this milestone

ymd = datetime.datetime.now().strftime("%y%m%d")
p: DQNAgent = p

##########################################

dc.seed = validation_game_seed

sesh = PlaySession(DigEnv(dc), p)
# p.eval()

sesh.play_game(1000)

sys.exit()
#########################################
