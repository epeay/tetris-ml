import os
import sys
from config import Hyperparameters, TMLConfig, load_config, hp


def before_tensorflow():
    module_name = "tensorflow"
    if module_name in sys.modules:
        raise ImportError(f"Module '{module_name}' has already been loaded.")

    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def make_model_player(cfg: TMLConfig, hp: Hyperparameters) -> "DQNAgent":

    from model import DQNAgent, DQNAgentConfig, TetrisCNN, TetrisCNNConfig

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


# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     tf.random.set_seed(seed)
