from dataclasses import dataclass
import gymnasium as gym
import json
import numpy as np
import random

from collections import deque
from numpy import ndarray as NDArray


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
from collections import deque
import os

import observer
from player import BasePlayer

from tetrisml.base import ActionContext, BaseEnv, EpisodeContext
import utils
from tetrisml import TetrisEnv, TetrisGameRecord, EnvStats, ModelAction, MinoShape
from tetrisml import Tetrominos
from tetrisml.display import GridRenderer, GridRenderConfig
from tetrisml.playback import GameFrame
import tetrisml.board

device = torch.device("cpu")
print(f"Using device {device}")


class ModelState:
    """
    Represents one state of the game board, exported for storage in the
    replay buffer, or flat file.
    """

    def __init__(self, board: NDArray, one_hot_width: int = None):

        if len(board.shape) != 2:
            raise ValueError("Board must be 2D")

        self.board: NDArray = board
        # One-hot encoding of the current mino
        # self.mino: NDArray = None

        self.one_hot_width: int | None = one_hot_width
        self.mino_queue_ids: list[int] = []

    @property
    def mino(self) -> MinoShape | None:
        if len(self.mino_queue_ids) == 0:
            return None

        return MinoShape(self.mino_queue_ids[0])

    def set_upcoming_minos(self, mino_queue: list[int | MinoShape]):
        self.mino_queue_ids = []

        for m in mino_queue:
            if isinstance(m, MinoShape):
                self.mino_queue_ids.append(m.shape_id)
            else:
                self.mino_queue_ids.append(m)

    def get_upcoming_minos(self) -> list[int]:
        return self.mino_queue_ids

    def set_mino_one_hot(self, total: int, index: int):
        """
        Deprecated. Use set_upcoming_minos.
        """
        print("Deprecated. Use set_upcoming_minos")
        return self.set_upcoming_minos([index])

    def read_mino_one_hot(self) -> int:
        print("Deprecated. Use get_upcoming_minos")

    def get_one_hot_data(self, width: int = None) -> NDArray:
        """
        Returns the mino one-hot encoding as a 1D array.
        """

        if width is not None:
            self.one_hot_width = width

        width = self.one_hot_width

        if self.one_hot_width is None:
            raise ValueError("One-hot width is not set")

        if len(self.mino_queue_ids) == 0:
            raise ValueError("Mino queue is empty")

        if max(self.mino_queue_ids) >= width:
            raise ValueError("One-hot width is too small")

        one_hots = []

        for m in self.mino_queue_ids:
            oh = np.zeros(width)
            oh[m] = 1
            one_hots.append(oh)

        return np.array(one_hots).flatten()

    def to_dict(self):
        return {"board": self.board.tolist(), "upcoming": self.mino_queue_ids}

    def get_linear_data(self, width: int = None) -> list[int]:
        return self.get_one_hot_data(width).tolist()

    def __repr__(self):
        pct_full = np.count_nonzero(self.board) / self.board.size

        queue_str = "<empty>"

        if len(self.mino_queue_ids) > 0:
            queue_str = ", ".join(
                [Tetrominos.shape_name(i) for i in self.mino_queue_ids]
            )

        GridRenderer.for_repr(self.board, self)
        return f"ModelState(board={pct_full:.2%} full), upcoming={queue_str})"

    def import_one_hots(self, one_hots: NDArray, width: int):
        """
        Unpacks linear data into a list of mino IDs. Overwrites the
        current mino queue.
        """
        if len(one_hots) % width != 0:
            raise ValueError("One-hots are not a multiple of the board width")

        items = len(one_hots) // width
        queue_ids = []
        for i in range(items):
            oh = one_hots[i * width : (i + 1) * width]
            if sum(oh) > 1:
                raise ValueError(f"One-hot encoding is not valid {oh}")
            elif sum(oh) == 1:
                queue_ids.append(np.argmax(oh))
            else:
                rest_of_data = one_hots[i * width :]
                if sum(rest_of_data) > 0:
                    raise ValueError(
                        f"Empty one-hot is followed by non-empty data {rest_of_data}"
                    )
                break

        self.mino_queue_ids = queue_ids

    @staticmethod
    def from_dict(d: dict):
        """
        num_tetrominos: The number of unique tetrominos in the game.
        """
        ret = ModelState(np.array(d["board"]))
        ret.mino_queue_ids = d["upcoming"]

        return ret


"""
cnn1:
    conv1: 32 filters, 3x3 kernel, stride 1, padding 1
    conv2: 64 filters, 3x3 kernel, stride 1, padding 1
    fc1: 128 + linear_layer_input_dim units
    fc2: action_dim units

cnn2:
    Adds a dropout layer

    conv1: 32 filters, 3x3 kernel, stride 1, padding 1
    conv2: 64 filters, 3x3 kernel, stride 1, padding 1
    fc1: 128 units
    dropout: configurable
    fc2: action_dim units
"""


@dataclass
class TetrisCNNConfig:
    linear_layer_input_dim: int
    model_id: str = None
    input_channels: int = 1
    board_height: int = 20
    board_width: int = 10
    action_dim: int = 40
    dropout_rate: float = 0.0  # Prevents dropout


class TetrisCNN(nn.Module):
    def __init__(
        self,
        config: TetrisCNNConfig,
    ):
        """
        Common param example:
            input_channels: 1
            board_height: 24  <- Does the model need to know about those top 4 rows?
            board_width: 10
            action_dim: 40  (10 columns * 4 rotations)
        """
        super().__init__()

        self.linear_data_dim = config.linear_layer_input_dim

        if config.model_id is None:
            raise ValueError("Model ID must be set")

        self.id = config.model_id
        self.conv1 = nn.Conv2d(
            config.input_channels, 32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            self.conv1.out_channels,
            self.conv1.out_channels * 2,  # 64
            kernel_size=3,
            stride=1,
            padding=1,
        )

        conv_output_size = (
            self.conv2.out_channels * config.board_height * config.board_width
        )

        self.fc1 = nn.Linear(conv_output_size + config.linear_layer_input_dim, 128)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc2 = nn.Linear(128, config.action_dim)

        self.intermediate_data = {}
        self.intermediate_gradients = {}

    def extend_linear_data(self, data: NDArray) -> NDArray:
        width = self.linear_data_dim
        # Pad the end with zeros
        if len(data) < width:
            data = np.pad(data, (0, width - len(data)), "constant")

        return data

    def forward(self, x_0, linear_layer_input=torch.tensor([])):
        # Expects board input shape (batch_size, channels, height, width)

        x_1 = torch.relu(self.conv1(x_0))
        self.intermediate_data["conv1_out"] = x_1.detach().clone()

        x_2 = torch.relu(self.conv2(x_1))
        self.intermediate_data["conv2_out"] = x_2.detach().clone()

        # Register hook to capture gradients during the backward pass
        x_2.register_hook(
            lambda grad: self.intermediate_gradients.update({"conv2": grad})
        )

        x = x_2.view(x_2.size(0), -1)  # Flatten the CNN output

        # Scale up linear layer input to match batch size
        if linear_layer_input.shape[0] != x.shape[0]:
            linear_layer_input = linear_layer_input.expand(x.shape[0], -1)

        # linear data should be shape (batch_size, linear_layer_input_dim)
        if len(linear_layer_input.shape) == 3 and linear_layer_input.shape[0] == 1:
            linear_layer_input = linear_layer_input.squeeze(1)

        x = torch.cat((x, linear_layer_input), dim=1)
        fc1_out = self.fc1(x)

        x = torch.relu(fc1_out)
        x = dropout_out = self.dropout(x)

        return self.fc2(x)


class AgentGameInfo:
    def __init__(self):
        self.agent_episode = None
        self.exploration_rate = None
        self.batch_episode = None
        self.batch_size = None


@dataclass
class StoredModelMeta:
    model_id: str
    board_height: int
    board_width: int
    unix_ts: int


class ModelCheckpoint(dict):
    def __init__(self):
        self.model_state: dict = None
        self.target_model_state: dict = None
        self.optimizer_state: dict = None
        self.replay_buffer: deque = None
        self.exploration: float = None
        self.episode: int = None
        self.meta: StoredModelMeta = None

    def __setattr__(self, key, value):
        """Class properties become dict key/value pairs"""
        self[key] = value
        super().__setattr__(key, value)


class ModelMemory:
    def __init__(self):
        self.state: ModelState = None
        self.action: ModelAction = None
        self.reward: float = None
        self.next_state: ModelState = None
        self.done: bool = None

    def to_tuple(self) -> tuple:
        return (self.state, self.action, self.reward, self.next_state, self.done)


class ModelParams(dict):
    def __init__(self):
        self.input_channels: int = None
        self.board_height: int = None
        self.board_width: int = None
        self.action_dim: int = None
        self.linear_data_dim: int = 0
        self.learning_rate: float = 0.001
        self.discount_factor: float = 0.99
        self.exploration_rate: float = 1.0
        self.exploration_decay: float = 0.99
        self.min_exploration_rate: float = 0.01
        self.replay_buffer_size: int = 10000
        self.batch_size: int = 64
        self.log_dir: str = None
        self.load_path: str = None
        self.model_id: str = None

    def __setattr__(self, key, value):
        """Class properties become dict key/value pairs"""
        self[key] = value
        super().__setattr__(key, value)


class ReplayBuffer(deque):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@dataclass
class DQNAgentConfig:
    # fmt: off
    input_channels        : int   = 1
    board_height          : int   = 20
    board_width           : int   = 10

    linear_data_dim       : int   = None
    action_dim            : int   = 40
    learning_rate         : float = 0.001
    discount_factor       : float = 0.99
    exploration_rate      : float = 1.0
    exploration_decay     : float = 0.999
    min_exploration_rate  : float = 0.01
    replay_buffer_size    : int   = 10000
    batch_size            : int   = 64
    log_dir               : str   = ""
    load_path             : str   = ""
    model_id              : str   = ""
    # fmt: on


class DQNAgent(BasePlayer):

    MODE_UNSET = 0
    MODE_EXPERT_LEARNING = 1

    def __init__(
        self,
        config: DQNAgentConfig,
        model: TetrisCNN = None,
        target_model: TetrisCNN = None,
    ):

        self.discount_factor = config.discount_factor
        self.exploration_rate = config.exploration_rate
        self.exploration_decay = config.exploration_decay
        self.min_exploration_rate = config.min_exploration_rate
        self.board_height = config.board_height
        self.board_width = config.board_width

        self.replay_buffer = deque(maxlen=config.replay_buffer_size)
        self.batch_size = config.batch_size
        self.reset_key = None
        self.num_rotations = 4
        # Show board state just before sending to model
        self.see_model_view = False
        self.mode = DQNAgent.MODE_UNSET

        self.pending_memory = ModelMemory()
        self.replays_counter = 0
        self.last_replay_episode = 0
        self.last_loss: float = 0

        self.action_stats = {}
        self.reset_action_stats()

        self.last_action_info = {}
        self.train = True

        if not len(config.model_id):
            config.model_id = utils.word_id()

        self.model = model
        self.target_model = target_model
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.game_records = []

        self.writer = (
            SummaryWriter(config.log_dir) if config.log_dir is not None else None
        )

        self.episode_memories = []

        self.agent_episode_count = 0

    def reset_action_stats(self):
        self.action_stats = {
            "predict": {
                "correct": 0,
                "incorrect": 0,
                "total": 0,
            },
            "guess": {
                "correct": 0,
                "incorrect": 0,
                "total": 0,
            },
        }

    def eval(self):
        """
        Locks the agent and underlying model from replaying/training. Also all
        exploration is disabled.
        """
        self.train = False
        self.exploration_rate = 0
        self.model.eval()
        self.target_model.eval()

    def save_model(self, abspath, overwrite=False):
        """
        Saves the model to the specified path. Does not include training
        parameters like exploration decay.
        """

        if not overwrite and os.path.exists(abspath):
            raise ValueError(f"Will not overwrite file: {abspath}")

        checkpoint = ModelCheckpoint()
        checkpoint.model_state = self.model.state_dict()
        checkpoint.target_model_state = self.target_model.state_dict()
        checkpoint.optimizer_state = self.optimizer.state_dict()
        checkpoint.replay_buffer = self.replay_buffer
        checkpoint.exploration = self.exploration_rate
        checkpoint.episode = self.agent_episode_count
        torch.save(checkpoint, abspath)

    def load_model(self, abspath):
        checkpoint: ModelCheckpoint = torch.load(abspath)

        self.model.load_state_dict(checkpoint.model_state)
        self.target_model.load_state_dict(checkpoint.target_model_state)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        self.replay_buffer = checkpoint.replay_buffer
        self.exploration_rate = checkpoint.exploration
        self.agent_episode_count = checkpoint.episode
        if self.agent_episode_count is None:  # For backward compatibility
            self.agent_episode_count = 0

    def log_game_record(self, game_record: TetrisGameRecord, envstats: EnvStats):

        if self.writer is None:
            print("WARNING: Not persisting game logs to disk")

        predict = 0
        guess = 0
        for was_predict in game_record.is_predict:
            if was_predict:
                predict += 1
            else:
                guess += 1

        predict_rate = int(predict / (predict + guess) * 10000) / 100

        r = game_record

        # Wrong place to modify the record object
        r.move_guesses = guess
        r.move_predictions = predict
        r.prediction_rate = predict_rate
        r.invalid_move_pct = r.invalid_moves / (r.moves + r.invalid_moves)
        r.avg_time_per_move = r.duration_ns / r.moves / 1000000000

        print(
            f"Episode {r.agent_info.batch_episode} of {r.agent_info.batch_size}. Agent run #{r.agent_info.agent_episode}"
        )
        print(f"Moves: {r.moves}")
        print(f"Invalid Moves: {r.invalid_moves}")
        print(f"Lines cleared: {r.lines_cleared}  ({str(r.cleared_by_size)})")
        print(f"Highest Reward: {max(r.rewards)}")
        print(f"Prediction Rate: {predict_rate} ({predict} of {predict+guess})")
        print(f"Duration: {r.duration_ns / 1000000000}")
        print(f"Agent Exploration Rate: {r.agent_info.exploration_rate}")
        print(f"AGENT Total Lines Cleared: {envstats.total_lines_cleared}")
        print(f"Predict Wins: {game_record.predict_wins}")
        if r.loss is not None:
            print(f"Loss {r.loss}")

        episode = r.agent_info.agent_episode

        if not self.writer:
            return

        self.writer.add_scalar("Episode/Total Moves", r.moves, episode)
        self.writer.add_scalar("Episode/% Invalid Moves", r.invalid_moves, episode)
        self.writer.add_scalar("Episode/Lines Cleared", r.lines_cleared, episode)
        self.writer.add_scalar(
            "Episode/Cumulative Reward", r.cumulative_reward, episode
        )
        self.writer.add_scalar("Episode/Prediction Rate", predict_rate, episode)
        self.writer.add_scalar("Episode/Duration", r.duration_ns / 1000000000, episode)
        self.writer.add_scalar(
            "Episode/Avg Time Per Move", r.avg_time_per_move, episode
        )
        self.writer.add_scalar("Episode/Predict Wins", r.predict_wins, episode)
        self.writer.add_scalar(
            "Agent/Total Lines Cleared", envstats.total_lines_cleared, episode
        )

        if r.loss is not None:
            self.writer.add_scalar("Episode/Loss", r.loss, episode)

        # Used to more easily identify runs that don't
        # have many episodes, for culling.
        self.writer.add_scalar("Agent/Episode", episode, episode)

    def save_game_records(self, filename="game_records.json"):
        with open(filename, "w") as f:
            json.dump([record.__dict__ for record in self.game_records], f)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(
        self,
        state: ModelState,
        action: ModelAction,
        reward,
        next_state: ModelState,
        done,
    ):
        state_dict = state.to_dict()
        next_state_dict = next_state.to_dict()

        self.replay_buffer.append((state_dict, action, reward, next_state_dict, done))

    def guess(self, m: MinoShape) -> ModelAction:
        """
        Generates a random action based on the action dimensions.
        If mino is specified, col will not cause the piece to go out of bounds.
        """
        rotation = np.random.choice(4)
        m = MinoShape(m.shape_id, rotation)
        valid_width = self.board_width - m.width
        col = np.random.choice(valid_width)  # range [0, value_width)

        return (col, rotation)

    def predict(self, state: ModelState) -> ModelAction:
        # (24, 10) -> (1, 1, 24, 10)
        # For (batch_size, channels, height, width)
        board = torch.FloatTensor(state.board)
        while len(board.shape) < 4:
            board = board.unsqueeze(0)

        linear_data = self.model.extend_linear_data(state.get_linear_data())
        linear_data = torch.FloatTensor(linear_data)
        # while len(linear_data.shape) < 3:
        #     linear_data = linear_data.unsqueeze(0)

        q_values = self.model(board, linear_data)
        action_index = torch.argmax(q_values).item()
        return (action_index // 4, action_index % 4)

    def choose_action(
        self, state: ModelState, m: MinoShape
    ) -> tuple[ModelAction, bool]:

        if random.random() < self.exploration_rate:
            return self.guess(m), False
        else:
            action_index = self.predict(state)
            return action_index, True

    def make_model_state(self, e: BaseEnv) -> ModelState:
        state = ModelState(
            e.board.export_board(), one_hot_width=Tetrominos.get_num_tetrominos()
        )
        state.set_upcoming_minos(e.mino_queue)
        return state

    def make_state_from_frame(self, frame: GameFrame) -> ModelState:
        state = ModelState(frame.board, one_hot_width=Tetrominos.get_num_tetrominos())
        state.set_upcoming_minos(frame.mino_queue)
        return state

    def make_memory_from_frame(
        self, frame: GameFrame, next_frame: GameFrame | None = None
    ) -> ModelMemory:
        ret = ModelMemory()
        state = self.make_state_from_frame(frame)
        ret.state = state
        ret.action = (frame.action_col, frame.action_rot)
        ret.next_state = None
        ret.done = True
        if next_frame is not None:
            ret.next_state = self.make_state_from_frame(next_frame)
            ret.done = False

        ret.reward = tetrisml.board.calculate_reward(ret.state.board)

        return ret

    def play(self, env: BaseEnv) -> ModelAction:

        self.last_action_info = {}
        lai = self.last_action_info

        self.pending_memory = ModelMemory()

        curr_state = self.make_model_state(env)
        action, is_prediction = self.choose_action(curr_state, env.get_current_mino())

        lai["shape_name"] = Tetrominos.shape_name(env.get_current_mino().shape_id)
        lai["column"] = action[0]
        lai["rotation"] = action[1]
        lai["is_prediction"] = is_prediction

        self.pending_memory.state = curr_state
        self.pending_memory.action = action

        return action

    def on_invalid_input(self, ctx: ActionContext):

        # State remains unchanged
        self.pending_memory.next_state = self.pending_memory.state
        self.pending_memory.reward = -1
        self.pending_memory.done = False  # TODO Should be influenced by the env.
        memory = self.pending_memory.to_tuple()
        print("Punishing invalid move")
        self.remember(*memory)

    def on_action_commit(
        self,
        env: TetrisEnv,
        ctx: ActionContext,
        memory: ModelMemory,
    ):

        next_state = self.make_model_state(env)
        self.pending_memory.next_state = next_state
        reward = env.calculate_reward()
        self.pending_memory.reward = reward
        self.pending_memory.done = ctx.ends_game
        memory = self.pending_memory.to_tuple()
        self.remember(*memory)

    def get_wandb_dict(self):
        ret = {
            "episode": self.agent_episode_count,
            "exploration_rate": self.exploration_rate,
            "replay_buffer_size": len(self.replay_buffer),
            "replayed_steps_count": self.replays_counter,
            "predict": {
                "1": self.action_stats["predict"]["correct"],
                "0": self.action_stats["predict"]["incorrect"],
                "total": self.action_stats["predict"]["total"],
            },
            "loss": self.last_loss,
        }

        if sum(self.action_stats["guess"].values()) > 0:
            ret["guess"] = {
                "1": self.action_stats["guess"]["correct"],
                "0": self.action_stats["guess"]["incorrect"],
                "total": self.action_stats["guess"]["total"],
            }

        return ret

    def get_debug_dict(self):

        predict_str = (
            "predicted" if self.last_action_info["is_prediction"] else "guessed"
        )

        guess_xofy = (
            self.action_stats["guess"]["correct"],
            self.action_stats["guess"]["correct"]
            + self.action_stats["guess"]["incorrect"],
        )

        predict_xofy = (
            self.action_stats["predict"]["correct"],
            self.action_stats["predict"]["correct"]
            + self.action_stats["predict"]["incorrect"],
        )

        guess_stats_str = f"{guess_xofy[0]} of {guess_xofy[1]}"
        if guess_xofy[1] > 0:
            guess_stats_str += f" ({int(guess_xofy[0]/guess_xofy[1]*100)}%)"

        predict_stats_str = f"{predict_xofy[0]} of {predict_xofy[1]}"
        if predict_xofy[1] > 0:
            predict_stats_str += f" ({int(predict_xofy[0]/predict_xofy[1]*100)}%)"

        return {
            "exploration_rate": int(self.exploration_rate * 100) / 100,
            "replay_buffer_size": len(self.replay_buffer),
            "replayed steps": self.replays_counter,
            "move was": predict_str,
            "guess success  ": guess_stats_str,
            "predict success": predict_stats_str,
            "last loss": self.last_loss,
        }

    def on_episode_start(self, env: BaseEnv):
        self.agent_episode_count += 1
        self.episode_memories = []

    def on_episode_end(self, ctx: ActionContext, e_ctx: EpisodeContext, env: TetrisEnv):

        correct = e_ctx.lines_cleared == 2

        modality = "predict" if self.last_action_info["is_prediction"] else "guess"
        outcome_str = "correct" if correct else "incorrect"
        self.action_stats[modality][outcome_str] += 1
        self.action_stats[modality]["total"] += 1

        # Should this be running every action? Every episode?
        loss = None
        if self.train:
            loss = self.replay()

        if self.train and loss is not None:
            self.decay_exploration_rate()

    def run(
        self,
        env: TetrisEnv,
        num_episodes=10,
        train=True,
    ):
        total_rewards = []
        target_update_interval = 10

        # Let's wait until the end of the game to
        # determine whether to store these states or not.

        for episode in range(num_episodes):
            self.agent_episode_count += 1

            # Capture the most recent game record.
            # TODO But do we not capture the final record?
            if env.record.moves > 0:
                self.game_records.append(env.record)

            board = env.reset()
            total_reward = 0
            done = False
            loss = None
            record_game = False

            while not done:
                self.play(env)

            # We're done
            env.close_episode()
            game_state = env.record

            ainfo = AgentGameInfo()
            ainfo.agent_episode = self.agent_episode_count
            ainfo.loss = loss

            # X of Y for this current execution run of the agent
            # Within the lifecycle of this method execution.
            ainfo.batch_episode = episode + 1 if train else episode
            ainfo.batch_size = num_episodes
            ainfo.exploration_rate = self.exploration_rate if train else 0
            env.record.agent_info = ainfo

            if train:
                self.decay_exploration_rate()

            record: TetrisGameRecord = env.record
            record.loss = loss
            self.log_game_record(record, env.stats)
            total_rewards.append(total_reward)

            if train and (episode % target_update_interval == 0):
                self.update_target_model()
                print("Updated target model")

        return total_rewards

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        minibatch = random.sample(self.replay_buffer, self.batch_size)
        composite_states, actions, rewards, composite_next_states, dones = zip(
            *minibatch
        )

        # Parse states from dicts
        composite_states = [ModelState.from_dict(s) for s in composite_states]
        composite_next_states = [ModelState.from_dict(s) for s in composite_next_states]

        # set linear data dim
        for s in composite_states:
            s.linear_data_dim = self.model.linear_data_dim

        # Unpack composite states
        boards = [s.board for s in composite_states]
        lds = [
            self.model.extend_linear_data(
                s.get_linear_data(Tetrominos.get_num_tetrominos())
            )
            for s in composite_states
        ]
        n_boards = [s.board for s in composite_next_states]
        n_lds = [
            self.model.extend_linear_data(
                s.get_linear_data(Tetrominos.get_num_tetrominos())
            )
            for s in composite_next_states
        ]

        # Ensure all states have consistent shapes
        boards = np.array(boards)
        lds = np.array(lds)
        n_boards = np.array(n_boards)
        n_lds = np.array(n_lds)

        boards = torch.FloatTensor(boards)
        lds = torch.FloatTensor(lds)
        n_boards = torch.FloatTensor(n_boards)
        n_lds = torch.FloatTensor(n_lds)

        actions = torch.LongTensor(
            [a[0] * 4 + a[1] for a in actions]
        )  # Ensure action is within valid range
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        if len(boards.shape) == 3:
            boards = boards.unsqueeze(1)  # Add channel dimension
            n_boards = n_boards.unsqueeze(1)

        # Get the q-values for all possible actions for all boards in the batch.
        # In:
        #   (batch_size, channels, height, width)
        #   (batch_size, linear_data_dim)
        # Out:
        #   (batch_size, action_dim)   action_dim = width * rotations
        #                              40 for a Nx10 board
        current_q_values = self.model(boards, lds)

        # Get the q-value of the chosen action for each board in the batch
        # Requires the actions to be of shape (batch_size, 1), in that there
        # are 64 scalar values as input, one per row.
        # e.g.: [[a1], [a2], [a3], ...]
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))

        # Drop from (batch_size, 1) to (batch_size,)
        current_q_values = current_q_values.squeeze(1)

        # Looking ahead to the next state:
        #   Calculate the q-values for each possible action. Shape (64, num_actions)
        #   Return the max q-value for each board in the batch. Shape (64,)
        #   [0] produces a tensor of shape (64, 1)
        max_next_q_values = self.target_model(n_boards, n_lds)
        max_next_q_values = max_next_q_values.max(1)[0]

        expected_q_values = rewards + (
            self.discount_factor * max_next_q_values * (1 - dones)
        )

        loss = self.loss_fn(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replays_counter += self.batch_size

        self.last_loss = loss.item()

        return loss

    def decay_exploration_rate(self):
        self.exploration_rate = max(
            self.min_exploration_rate, self.exploration_rate * self.exploration_decay
        )


class ModelPlayer(BasePlayer):

    def __init__(self, model: TetrisCNN):
        self.model: TetrisCNN = model
        self.action_stats = observer.ActionTracker()

    def make_model_state(self, e: BaseEnv) -> ModelState:
        """
        TODO Duplicate code from DQNAgent. Cleanup.
        """
        state = ModelState(e.board.export_board())
        state.set_mino_one_hot(
            Tetrominos.get_num_tetrominos(), e.get_current_mino().shape_id
        )
        return state

    def get_debug_dict(self):
        return self.get_wandb_dict()

    def get_wandb_dict(self):

        p_stats = self.action_stats.actions[observer.ActionTracker.T_PREDICT]
        ret = {
            "is_prediction": True,
            "stats": {
                "predict": {
                    "0": p_stats["0"],
                    "1": p_stats["1"],
                    "total": p_stats["total"],
                },
            },
        }
        print("Printing wandb dict")
        return ret

    def predict(self, state: ModelState) -> ModelAction:
        """
        TODO Duplicate code from DQNAgent. Cleanup.
        """
        # (24, 10) -> (1, 1, 24, 10)
        # For (batch_size, channels, height, width)
        board = torch.FloatTensor(state.board)
        while len(board.shape) < 4:
            board = board.unsqueeze(0)

        linear_data = torch.FloatTensor(state.get_linear_data())

        q_values = self.model(board, linear_data)
        action_index = torch.argmax(q_values).item()
        return (action_index // 4, action_index % 4)

    def play(self, env: TetrisEnv) -> tuple[ModelAction, dict]:
        state = self.make_model_state(env)
        action = self.predict(state)
        return action

    def on_action_commit(self, e: BaseEnv, action: ModelAction, done: bool):
        super().on_action_commit(e, action, done)
        reward = e.calculate_reward()
        correct = reward == 2
        self.action_stats.register_action(observer.ActionTracker.T_PREDICT, correct)

    @staticmethod
    def from_checkpoint(checkpoint_path: str, config: dict) -> "ModelPlayer":
        cp: ModelCheckpoint = torch.load(checkpoint_path)
        m = TetrisCNN(
            cp.model_id,
            cp.input_channels,
            cp.board_height,
            cp.board_width,
            cp.action_dim,
            cp.linear_data_dim,
        )
        m.model.load_state_dict(cp.model_state)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
