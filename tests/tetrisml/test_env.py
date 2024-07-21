import pytest

from tetrisml.env import MinoBag, TetrisEnv
from tetrisml.tetrominos import Tetrominos
from tetrisml.minos import MinoShape


@pytest.fixture
def mino_bag():
    tiles = [0, 1, 2, 3, 4, 5]
    seed = 42
    maxlen = 5
    return MinoBag(tiles, seed, maxlen)

def test_initial_population(mino_bag: MinoBag):
    assert len(mino_bag) == mino_bag.maxlen
    for item in mino_bag:
        assert item in mino_bag.tiles
 
def test_popleft(mino_bag: MinoBag):
    initial_len = len(mino_bag)
    initial_copy = list(mino_bag)  # Convert to list to copy
    item = mino_bag.popleft()
    assert item == initial_copy[0]
    assert initial_copy[1:] == list(mino_bag)[:-1]
    assert len(mino_bag) == initial_len

def test_pull(mino_bag: MinoBag):
    initial_len = len(mino_bag)
    item = mino_bag.pull()
    assert len(mino_bag) == initial_len
    assert item in mino_bag.tiles

def test_str(mino_bag: MinoBag):
    assert str(mino_bag).startswith("MinoBag(")



@pytest.fixture
def env():
    return TetrisEnv.tetris()

def test_TetrisEnv(env: TetrisEnv):
    assert env.board_height == 20
    assert env.board_width == 10
    assert env.piece_bag == Tetrominos.std_bag


def test_short_game(env: TetrisEnv):

    env.current_mino = MinoShape(Tetrominos.I, 1)  # Tall I

    for i in range(9):
        # Drop tall I's across the board
        env.current_mino = MinoShape(Tetrominos.I, i % 4) # rot shouldn't matter
        env.step((i, 1))
        assert sum([sum(x) for x in env.board.board]) == 4 * (i + 1)

    # Drop the last I to clear the board
    env.current_mino = MinoShape(Tetrominos.I)
    env.step((9, 1))
    assert sum([sum(x) for x in env.board.board]) == 0




