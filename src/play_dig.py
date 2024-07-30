import datetime
import os
import pprint
import random
import sys

import wandb
from model import DQNAgent
from tetrisml.dig import DigBoard, DigEnv
from tetrisml.env import PlaySession, TetrisEnv, E_MINO_SETTLED
from tetrisml.minos import MinoShape
from tetrisml.tetrominos import Tetrominos

from player import CheatingPlayer, RandomPlayer
from config import load_config, hp
import utils

wandb.require("core")

ENV_SEED = ""
model_id = utils.word_id()

config = load_config()

hyper = hp.copy()
hyper["env"]["git_hash_short"] = config.git_short
hyper["env"]["unix_ts"] = datetime.datetime.now().timestamp()
hyper["game"]["seed"] = model_id


wandb.init(
    project=hyper["env"]["project_name"],
    config=hyper,
)

h = 10
w = 5

p = None
input_channels = 1
action_dim = 4 * w  # 4 rotations
linear_data_dim = Tetrominos.get_num_tetrominos()

e = DigEnv(seed=model_id)


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

if p is None:
    p = make_model_player()
sesh = PlaySession(e, p)

data = {}
data["training"] = {}
data["eval"] = {}
data["eval_from_load"] = {}

import time

start = time.time()

validation_game_seed = "whiff-adorn-1"  # hardcoded for this milestone

while sesh.player.exploration_rate > 0.1:
    sesh.play_game(100, render=False)

print("Trained! Elapsed time:", time.time() - start)
time.sleep(0.3)
p: DQNAgent = p
save_path = os.path.join(config.model_storage_dir, f"{p.model.id}.pth")
p.save_model(save_path)
print(f"Model saved to {save_path}")
data["training"] = p.action_stats
p.reset_action_stats()

##########################################

sesh = PlaySession(DigEnv(seed=validation_game_seed), p)
p.eval()

sesh.play_game(1000)
data["eval"] = p.action_stats

sys.exit()
#########################################

load_path = save_path
# load_path = os.path.join(config.model_storage_dir, "eager-piano.pth")

# Load the model and play some games
p = make_model_player()
p.load_model(load_path)
print(f"Model loaded from {load_path}")

# No training. All predictions.
p.eval()
sesh = PlaySession(DigEnv(), p)
sesh.play_game(1000)
data["eval_from_load"] = p.action_stats

pprint.pprint(data)


print("Done")
