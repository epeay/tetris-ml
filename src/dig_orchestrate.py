import datetime
import os
import pprint
import random
import sys
import time

import setup

import wandb

from model import DQNAgent, DQNAgentConfig, TetrisCNN, TetrisCNNConfig
from tetrisml.dig import DigBoard, DigEnv, DigEnvConfig
from tetrisml.env import PlaySession, TetrisEnv
from tetrisml.minos import MinoShape
from tetrisml.tetrominos import Tetrominos
from tetrisml.playback import GameFrameCollection

from player import CheatingPlayer, RandomPlayer
from config import Hyperparameters, TMLConfig, load_config, hp

wandb.require("core")
config: TMLConfig = load_config()  # Generates run id
hp: Hyperparameters = Hyperparameters()
hp.model.action_dim = hp.board.width * 4  # 4 mino rotations
hp.model.linear_data_dim = 4 * Tetrominos.get_num_tetrominos()
hp.game.seed = config.model_id
hp.model.dropout_rate = 0.0

### Config finalized

wandb.init(
    project=config.project_name,
    config=hp,
    name=config.run_id,
)

e = DigEnv(
    DigEnvConfig(
        board_height=hp.board.height,
        board_width=hp.board.width,
        seed=hp.game.seed,
        solution_depth=2,
    )
)


sesh = PlaySession(e, CheatingPlayer())
sesh.play_game(100)

import tetrisml.board

winner = 0
total = 0

for g in sesh.session_games:
    total += 1
    last_frame = g.frames[-1]
    if last_frame.intermediate_board is not None:
        last_reward = tetrisml.board.calculate_reward(last_frame.intermediate_board)
        if last_reward == 2:
            winner += 1


print(f"Winner: {winner}/{total}")


mp = setup.make_model_player(config, hp)
mp: DQNAgent = mp


game = sesh.session_games[20]

memory = None
if len(game.frames) > 2:
    memory = mp.make_memory_from_frame(game.frames[1], game.frames[2])
else:
    memory = mp.make_memory_from_frame(game.frames[1])


sys.exit()


p = None

if p is None:
    p = setup.make_model_player(config, hp)
sesh = PlaySession(e, p)

start = config.unix_ts
validation_game_seed = "whiff-adorn-1"  # hardcoded for this milestone
# sesh.play_game(200)

while p.exploration_rate > 0.1:
    sesh.play_game(100)
    print()
    print("Exploration rate:", p.exploration_rate)
    print("Loss:", p.last_loss)

print("Trained! Elapsed time:", time.time() - start)


def make_histograms(games: list[GameFrameCollection]):
    from game_vis import create_board_image

    for gi, g in enumerate(games):
        boards = []
        for fi, f in enumerate(g):
            if f.intermediate_board is not None:
                boards.append(f.intermediate_board)
            boards.append(f.board)

        create_board_image(boards).save(f"game-histogram-{gi+1}.png")


time.sleep(0.3)


p: DQNAgent = p
save_path = os.path.join(config.model_storage_dir, f"{config.model_id}.pth")
p.save_model(save_path)
print(f"Model saved to {save_path}")


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
