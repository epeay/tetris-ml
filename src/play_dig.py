from tetrisml.dig import DigBoard, DigEnv
from tetrisml.env import PlaySession, TetrisEnv, E_MINO_SETTLED
from tetrisml.minos import MinoShape
from tetrisml.tetrominos import Tetrominos
import time

from player import CheatingPlayer, RandomPlayer


e = DigEnv()
p = None
input_channels = 1
action_dim = 4 * e.board_width  # 4 rotations
linear_data_dim = Tetrominos.get_num_tetrominos()


def make_model_player():
    # Keeping this import separate because it's not always needed
    # and tensorflow is slow to import
    from model import DQNAgent

    p = DQNAgent(
        input_channels,
        e.board_height,
        e.board_width,
        action_dim,
        linear_data_dim=linear_data_dim,
    )

    return p


# p = RandomPlayer()

if p is None:
    p = make_model_player()
sesh = PlaySession(e, p)
e.reset()

sesh.play_game(100)
