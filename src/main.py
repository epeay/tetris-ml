import datetime
import gymnasium as gym
import json
import numpy as np
import os
import sys
import yaml

from tetrisml import *
from model import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
# Verify TensorFlow is using CPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

"""
Episode = One tetris game
mino = Tetromino
"""

WORKSPACE_ROOT = os.path.join(os.path.expanduser("~"), "source", "tetris-ml")
class TMLConfig(dict):
    def __init__(self):
        self.workspace_dir:str = os.path.normpath(WORKSPACE_ROOT)
        self.storage_root:str = os.path.join(WORKSPACE_ROOT, "storage")
        self.tensorboard_log_dir:str = os.path.join(self.storage_root, "tensor-logs")

    def __setattr__(self, key, value):
        """Class properties become dict key/value pairs"""
        self[key] = value
        super().__setattr__(key, value)

    def __getattr__(self, key):
        return self[key]



def load_config():
    # Create workspace directory
    config = TMLConfig()

    # Load ../config.yaml
    config_path = os.path.join(os.getcwd(), "config.yaml")
    with open(config_path, 'r') as stream:
        try:
            config.update(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)


    os.makedirs(config.workspace_dir, exist_ok=True)
    os.makedirs(config.storage_root, exist_ok=True)
    os.makedirs(config.tensorboard_log_dir, exist_ok=True)

    return config


config = load_config()
game_logs:list[GameHistory] = []


def mcts(game_logs:list[GameHistory]=None):
    env = TetrisEnv([Tetrominos.O, Tetrominos.USCORE, Tetrominos.DOT])

    for _ in range(5):
        env.reset()
        history:GameHistory = GameHistory()
        print(f"Starting game {history.id}")
        history.seed = env.random_seed
        history.bag = env.piece_bag

        move = 0
        while True:
            move += 1
            piece = env.current_piece
            possibilities = []
            best_reward = -np.inf
            # TODO: Some minos don't need four rotations
            for i in range(4):
                possibilities += (find_possible_moves(env, MinoShape(piece.shape, i)))

            highest_reward_choices = []

            for p in possibilities:
                if p.reward == best_reward:
                    highest_reward_choices.append(p)
                if p.reward > best_reward:
                    best_reward = p.reward
                    highest_reward_choices = [p]
            
            best_choice:MinoPlacement = np.random.choice(highest_reward_choices)
            env.step((best_choice.bl_coords[1]-1, best_choice.shape.shape_rot))
            history.placements.append(best_choice)
            print("--------------------------")
            print(f"Move {move}")
            print(f"Best Choice: {best_choice}")
            env.render()
            print("--------------------------")


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
                    print("Tetris!!!!!!!!!!")
                    print("Tetris!!!!!!!!!!")
                    print("Tetris!!!!!!!!!!")
                    print("Tetris!!!!!!!!!!")

                break

        game_logs.append(history)




def save_game_logs(game_logs:list[GameHistory], path:str="game_logs.json"):
    ret = {"games": {}}

    for game in game_logs:
        ret["games"][game.id] = game.to_jsonable()

    print(f"Saving game logs to {path}")
    with open(path, "w") as outfile:
        json.dump(ret, outfile, indent=4)


try:
    mcts(game_logs=game_logs)
except KeyboardInterrupt:
    print("Keyboard interrupt. Saving game logs.")

if len(game_logs) > 0:
    file_ts = game_logs[0].timestamp.strftime("%y%m%d_%H%M%S")
    save_game_logs(game_logs, f"game_logs_{file_ts}.json")

sys.exit()


def main():
  # Example usage
  env = TetrisEnv()
  env.piece_bag = Tetrominos.std_bag
  state = env.reset()

  done = False
  loop_limit = 2
  loop = 0
  while not done and loop < loop_limit:
      action = env.action_space.sample()  # Random action for demonstration
      next_state, reward, done, info = env.step(action)
      env.board.render()
      print(f"Reward: {reward}, Done: {done}")
      print(info)
      print("----------------------")
      loop += 1

  print(env.record.__dict__)


run_comment:str = "CNN-cpu-all-Os-datetime-test"
persist_logs = config.persist_logs

show_board_before_running_model = False # @param {type:"boolean"}

save_test = False

# Initialize Tetris environment
env = TetrisEnv(piece_bag=[Tetrominos.O])


input_channels = 1
board_height = 24   # 20 for playfield, 4 for staging next piece
board_width = 10
action_dim = 40  # 4 rotations * 10 columns



log_dir = None
if persist_logs:
    # YYMMDD-HHMMSS
    current_time = datetime.now().strftime('%y%m%d_%H%M%S')
    log_dir = os.path.join(config.tensorboard_log_dir, f'{current_time}-{run_comment}')
agent = DQNAgent(input_channels, board_height, board_width, action_dim, log_dir=log_dir)



# if save_test:
#     agent.run(env, 10, train = True)
#     print(f"Replay buffer size: {len(agent.replay_buffer)}")
#     print(f"Exploration rate: {agent.exploration_rate}")
#     print(f"Agent episode count: {agent.agent_episode_count}")
#     agent.save_model("/content/drive/MyDrive/tensor-logs/models/test.pth")

# else:


#     print(f"Replay buffer size: {len(agent.replay_buffer)}")
#     print(f"Exploration rate: {agent.exploration_rate}")
#     print(f"Agent episode count: {agent.agent_episode_count}")
#     print("--------------------------")
#     print("loading model")
#     agent.load_model("/content/drive/MyDrive/tensor-logs/models/test.pth")
#     print(f"Replay buffer size: {len(agent.replay_buffer)}")
#     print(f"Exploration rate: {agent.exploration_rate}")
#     print(f"Agent episode count: {agent.agent_episode_count}")

# sys.exit()

# print(len(agent.replay_buffer))
# agent.load_model("/content/drive/MyDrive/tensor-logs/models/test.pth")
# sys.exit()

num_episodes = 10
target_update_interval = 10

training_tracker = []

model_save_dir = "/content/drive/MyDrive/tensor-logs/models"


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



training_data = json.load(open(config.workspace_dir + "/game_logs_240714_214823.json"))


agent.run(env, 5, train = True)

save_interval = 500

if keep_training(agent):
    agent.run(env, 500, train = True)
else:
    print("Training SUCCEEDED!!!!!")
    # filename = f"tetris_ep{agent.agent_episode_count}_TRAINED.pth"
    # full_path = os.path.join(model_save_dir, filename)
    # torch.save(agent.model.state_dict(), full_path)
    # print(f"Saved model to {full_path}")




# agent.train(env, 10)








