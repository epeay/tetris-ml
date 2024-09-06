import pytest
import sys
import numpy as np
from io import StringIO


from tetrisml.board import TetrisBoard
from tetrisml.env import MinoBag, TetrisEnv
from tetrisml.tetrominos import Tetrominos
from tetrisml.minos import MinoShape


@pytest.fixture
def mino_bag():
    tiles = [0, 1, 2, 3, 4, 5]
    seed = 42
    maxlen = 5
    return MinoBag(tiles, seed, maxlen)


@pytest.fixture
def env():
    return TetrisEnv.tetris()


@pytest.fixture
def sml():
    return TetrisEnv.smoltris()


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


def test_TetrisEnv(env: TetrisEnv):
    assert env.board_height == 20
    assert env.board_width == 10
    assert env.piece_bag.tiles == Tetrominos.std_bag


def test_short_game(env: TetrisEnv):

    pytest.skip("Needs refactoring")

    env.current_mino = MinoShape(Tetrominos.I, 1)  # Tall I

    for i in range(9):
        # Drop tall I's across the board
        env.current_mino = MinoShape(Tetrominos.I, i % 4)  # rot shouldn't matter
        env.step((i, 1))
        assert sum([sum(x) for x in env.board.board]) == 4 * (i + 1)

    # Drop the last I to clear the board
    env.current_mino = MinoShape(Tetrominos.I)
    env.step((9, 1))
    assert sum([sum(x) for x in env.board.board]) == 0


def test_render_last_action():
    board = np.zeros((4, 4), dtype=int)
    mino = MinoShape(Tetrominos.O)
    lcoords = (1, 3)

    expected_output = (
        "== Test ==\n"  # fmt: skip
        "_ _ _ _ \n"
        "_ _ _ _ \n"
        "_ _ ■ ■ \n"
        "_ _ ■ ■ \n"
    )

    # Redirect stdout to capture print statements
    captured_output = StringIO()
    sys.stdout = captured_output

    TetrisBoard.render_last_action(board, mino, lcoords, title="Test")

    # Reset redirect.
    sys.stdout = sys.__stdout__

    assert captured_output.getvalue(), expected_output


def test_render_last_action_matrix():

    mino = MinoShape(Tetrominos.S)
    env = TetrisEnv.smoltris()
    env.board.place_shape(mino, (1, 1))

    expected = np.zeros((env.board_height + 4, env.board_width), dtype=int)

    # Human readable "S" mino pattern
    s_mino_pattern = np.array(
        [
            [0, 2, 2],
            [2, 2, 0],
        ]
    )

    # ...flipped to game-appropriate orientation
    s_mino_pattern = np.flipud(s_mino_pattern)

    # Insert expected into big_board
    expected[0:2, 0:3] = s_mino_pattern

    actual = TetrisBoard.render_last_action(
        env.board.board, mino, (1, 1), title="Test", return_matrix=True
    )

    # Assert matrices match
    assert actual.shape == expected.shape
    assert all([all(a == b) for a, b in zip(expected, actual)])
