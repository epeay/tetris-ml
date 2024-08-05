from dataclasses import dataclass, field
import datetime
import os
import yaml  # type: ignore
import utils
from typing import Optional

# ../
WORKSPACE_ROOT = os.path.dirname(os.path.realpath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(WORKSPACE_ROOT, ".."))


@dataclass
class ModelHP:
    arch: str = "cnn2"  # Not a tunable
    input_channels: int = 1  # Not a tunable
    dropout_rate: float = 0.1
    linear_data_dim: int = 0  # Not a tunable. Must be overwritten at runtime.
    action_dim: int = 0  # Not a tunable. Must be overwritten at runtime.


@dataclass
class BoardHP:
    height: int = 20
    width: int = 10


@dataclass
class GameHP:
    type: str = "dig"
    seed: Optional[int] = None


@dataclass
class AgentHP:
    exploration_rate: float = 1.0
    exploration_decay: float = 0.999
    learning_rate: float = 0.01
    batch_size: int = 64
    replay_buffer_size: int = 1000


@dataclass
class Hyperparameters:
    model: ModelHP = ModelHP()
    agent: AgentHP = AgentHP()
    board: BoardHP = BoardHP()
    game: GameHP = GameHP()


hp = Hyperparameters()


@dataclass
class TMLConfig:
    unix_ts: float
    run_id: str
    model_id: str
    workspace_dir: str
    storage_root: str
    tensorboard_log_dir: str
    model_storage_dir: str
    persist_logs: bool = False
    git_short: str = field(default_factory=utils.get_git_hash)
    git_pristine: bool = False
    project_name: str = "tetris-ml"


def load_config() -> TMLConfig:

    unix_ts = int(datetime.datetime.now().timestamp())

    dt = datetime.datetime.fromtimestamp(unix_ts)
    ymd = dt.strftime("%y%m%d")

    word_id = utils.word_id()
    run_id = f"{ymd}-{word_id}"

    # fmt: off
    config = TMLConfig(
        run_id                  = run_id,
        model_id                = run_id,
        workspace_dir           = WORKSPACE_ROOT,
        storage_root            = os.path.join(WORKSPACE_ROOT, "storage"),
        tensorboard_log_dir     = os.path.join(WORKSPACE_ROOT, "storage", "tensor-logs"),
        model_storage_dir       = os.path.join(WORKSPACE_ROOT, "storage", "models"),
        unix_ts                 = unix_ts,
    )
    # fmt: on

    os.makedirs(config.workspace_dir, exist_ok=True)
    os.makedirs(config.storage_root, exist_ok=True)
    os.makedirs(config.tensorboard_log_dir, exist_ok=True)
    os.makedirs(config.model_storage_dir, exist_ok=True)

    return config
