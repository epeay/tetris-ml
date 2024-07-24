import datetime
import json
import numpy as np
import os
import sys
import yaml

import matplotlib.pyplot as plt

from config import load_config, TMLConfig
from tetrisml import *
from model import *
from tetrisml.gamesets import GameRuns
import utils
from cheating import *
from viz import *
import time
from tetrisml.env import PlaySession
from player import CheatingPlayer, PlaybackPlayer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# Verify TensorFlow is using CPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

"""
Episode = One tetris game
mino = Tetromino
"""


config: TMLConfig = load_config()
game_logs: list[GameHistory] = []


def mcts(env: TetrisEnv, episodes: int = 10, game_logs: list[GameHistory] = None):

    for _ in range(episodes):
        env.reset()
        history: GameHistory = GameHistory()
        print(f"Starting game {history.id}")
        history.seed = env.random_seed
        history.bag = env.piece_bag

        move = 0
        while True:
            move += 1
            piece = env.current_mino
            possibilities = []
            best_reward = -np.inf
            # TODO: Some minos don't need four rotations
            for i in range(4):
                possibilities += find_possible_moves(env, MinoShape(piece.shape_id, i))

            highest_reward_choices = []

            for p in possibilities:
                if p.reward == best_reward:
                    highest_reward_choices.append(p)
                if p.reward > best_reward:
                    best_reward = p.reward
                    highest_reward_choices = [p]

            best_choice: MinoPlacement = np.random.choice(highest_reward_choices)
            env.step((best_choice.bl_coords[1] - 1, best_choice.shape.shape_rot))
            history.placements.append(best_choice)
            env.render()

            if env.board.board[-4:].any():
                env.close_episode()
                history.record = env.record

                print("Game Over")
                print(f"GAME ID: {history.id}")
                print(f"Final Reward: {env.record.cumulative_reward}")
                print(f"Lines Cleared: {env.record.lines_cleared}")
                print(f"Invalid Moves: {env.record.invalid_moves}")
                print(f"Clears by Size: {env.record.cleared_by_size}")
                print(f"Duration: {env.record.duration_ns / 1000000000}")
                print(f"Moves: {env.record.moves}")
                print(f"Game Seed: {env.random_seed}")

                if env.record.cleared_by_size[4] > 0:
                    print("Tetris!!!!!!!!!!")

                # I'm about to
                break

        game_logs.append(history)


def save_game_logs(game_logs: list[GameHistory], path: str = "game_logs.json"):
    ret = {"games": {}}

    for game in game_logs:
        ret["games"][game.id] = game.to_jsonable()

    print(f"Saving game logs to {path}")
    with open(path, "w") as outfile:
        json.dump(ret, outfile, indent=4)


def run_mcts(env: TetrisEnv, episodes: int = 10):
    mcts(env, episodes=episodes, game_logs=game_logs)
    if len(game_logs) > 0:
        file_ts = game_logs[0].timestamp.strftime("%y%m%d_%H%M%S")
        save_game_logs(game_logs, f"game_logs_{file_ts}.json")
    sys.exit()


e = TetrisEnv.smoltris()

input_channels = 1
action_dim = 4 * e.board_width  # 4 rotations
linear_data_dim = Tetrominos.get_num_tetrominos()

# YYMMDD-HHMMSS
current_time = datetime.now().strftime("%y%m%d_%H%M%S")
model_id = utils.word_id()
log_dir = os.path.join(
    config.tensorboard_log_dir, f"{current_time}-{model_id}-{config.slug}"
)

use_log_dir = log_dir if config.persist_logs else None
agent = DQNAgent(
    input_channels,
    e.board_height,
    e.board_width,
    action_dim,
    linear_data_dim=linear_data_dim,
    log_dir=use_log_dir,
    model_id=model_id,
)


# agent.run(e, 1)

# run_mcts(smoltris, episodes=100) # and exit()

# load_path = os.path.join(config.workspace_dir, "storage", "models", "funnyhouse-trained-smoltris.pth")
# agent.load_model(load_path)

# while smoltris.stats.total_lines_cleared < 1000:
#     agent.run(smoltris, 10)

target_update_interval = 10


def keep_training(agent):
    """
    Run hueristics around recent game performance, and continue
    training if necessary.
    """

    if agent.agent_episode_count >= 10000:
        return False

    records = agent.game_records[-50:]

    avg_line_clears_per_game = np.average([r.lines_cleared for r in records])
    avg_moves_per_game = np.average([r.moves for r in records])
    avg_invalid_move_pct = np.average([r.invalid_move_pct for r in records])

    criteria = []
    criteria.append(avg_line_clears_per_game > 10)
    criteria.append(avg_moves_per_game > 80)
    criteria.append(avg_invalid_move_pct < 1)

    print(f"Evaluating hueristics over {len(records)} games")
    print(f"Avg Line Clears: {avg_line_clears_per_game}")
    print(f"Avg Moves: {avg_moves_per_game}")
    print(f"Avg Invalid Move %: {avg_invalid_move_pct}")

    if np.all(criteria):
        return False

    # Resetting progress
    if agent.exploration_rate < 0.1:
        agent.exploration_rate = 0.8

    return True


def run_from_playback(path: str):
    num_games = len(playback)
    # episode count doesn't matter when playback is specified
    agent.run(e, 10, playback_list=[playback[0]])


e = TetrisEnv.smoltris()

# playback_path = os.path.join(
#     config.workspace_dir, "game_logs_240718_215902_expert_smoltris.json"
# )
# p = PlaybackPlayer.from_file(playback_path)

p = CheatingPlayer()
sesh = PlaySession(e, p)

# sesh.events.register(E_STEP_COMPLETE, lambda: print(".", end=""))


def on_mino_settled():
    print("")
    e.render()
    time.sleep(0.3)
    print("")


sesh.env.events.register(E_MINO_SETTLED, on_mino_settled)
sesh.play(10)


# playback_path = os.path.join(config.workspace_dir, "game_logs_240718_215902_expert_smoltris.json")
# print(playback_path)
# run_from_playback(playback_path)


# env = TetrisEnv(piece_bag=ODU, board_height=10, board_width=5)
# while env.stats.total_lines_cleared < 10000:
#     agent.run(env, 10)

# save_location = os.path.join(
#     config.model_storage_dir,
#     f"{agent.model.id}.pth")
# agent.save_model(save_location)


# sets = GameRuns(config.workspace_dir)
# games = sets.load("game_logs_240717_144432")
# sets.cumulative_reward_plot(games)
