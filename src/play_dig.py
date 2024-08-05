import datetime
import os
import pprint
import random
import sys
import time

from tetrisml.playback import GameFrameCollection

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Must come after setting these vars
import tensorflow as tf  # type: ignore

import wandb
from model import DQNAgent, DQNAgentConfig, TetrisCNN, TetrisCNNConfig
from tetrisml.dig import DigBoard, DigEnv, DigEnvConfig
from tetrisml.env import PlaySession, TetrisEnv
from tetrisml.minos import MinoShape
from tetrisml.tetrominos import Tetrominos

from player import CheatingPlayer, RandomPlayer
from config import Hyperparameters, TMLConfig, load_config, hp

assert len(tf.config.experimental.list_physical_devices("GPU")) == 0

wandb.require("core")
config: TMLConfig = load_config()
hp: Hyperparameters = Hyperparameters()

hp.game.seed = config.model_id
hp.model.dropout_rate = 0.0

wandb_config = {
    "cfg": config,
    "hp": hp,
}
wandb.init(
    project=config.project_name,
    config=wandb_config,
    name=config.run_id,
)

# Some config values are set at runtime
hp.model.action_dim = hp.board.width * 4  # 4 mino rotations
hp.model.linear_data_dim = Tetrominos.get_num_tetrominos()

wandb.config.update(wandb_config, allow_val_change=True)

e = DigEnv(
    DigEnvConfig(
        board_height=hp.board.height,
        board_width=hp.board.width,
        seed=hp.game.seed,
    )
)

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


# e = TetrisEnv(Tetrominos.std_bag, 20, 10)
# p = CheatingPlayer()

if p is None:
    p = make_model_player(config, hp)
sesh = PlaySession(e, p)

import time

start = config.unix_ts


validation_game_seed = "whiff-adorn-1"  # hardcoded for this milestone

p: CheatingPlayer = sesh.player


sesh.play_game(2)


while p.exploration_rate > 0.1:
    sesh.play_game(100, render=False)
    print()
    print("Exploration rate:", p.exploration_rate)
    print("Loss:", p.last_loss)

print("Trained! Elapsed time:", time.time() - start)

from game_vis import *

sg: list[GameFrameCollection] = sesh.session_games
for gi, g in enumerate(sg):
    boards = []
    for fi, f in enumerate(g):
        if f.intermediate_board is not None:
            boards.append(f.intermediate_board)
        boards.append(f.board)

    # create_board_image(boards).save(f"game-histogram-{gi+1}.png")


time.sleep(0.3)


# p: DQNAgent = p
# save_path = os.path.join(config.model_storage_dir, f"{config.model_id}.pth")
# p.save_model(save_path)
# print(f"Model saved to {save_path}")


# p.load_model(save_path)

sys.exit()


data["training"] = p.action_stats
p.reset_action_stats()

##########################################

dc.seed = validation_game_seed

sesh = PlaySession(DigEnv(dc), p)
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
