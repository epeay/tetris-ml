from tetrisml.board import TetrisBoard, clear_full_lines
from tetrisml.minos import MinoShape
from tetrisml.tetrominos import *
import numpy as np


def test_from_ascii():
    ascii = [  # fmt: skip
        "X X",
        " X ",
        "XXX",
    ]

    board = TetrisBoard.from_ascii(ascii)
    expected = np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]])
    assert np.array_equal(board.board, expected)

    # Test padding
    board = TetrisBoard.from_ascii(ascii, h=20, w=10)
    expected = np.zeros((20, 10), dtype=int)
    expected[0:3, 0:3] = np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]])
    assert np.array_equal(board.board, expected)


def test_find_logical_BL_coords():
    ascii = [  # fmt: skip
        "X X",
        " X ",
        "XXX",
    ]

    side_i = MinoShape(Tetrominos.I, 0)
    tall_i = MinoShape(Tetrominos.I, 1)

    board = TetrisBoard.from_ascii(ascii, h=20, w=10)
    mino = tall_i

    assert board.find_logical_BL_coords(tall_i, 0) == (4, 1)
    assert board.find_logical_BL_coords(tall_i, 1) == (3, 2)
    assert board.find_logical_BL_coords(tall_i, 2) == (4, 3)
    assert board.find_logical_BL_coords(tall_i, 3) == (1, 4)
    assert board.find_logical_BL_coords(tall_i, 9) == (1, 10)

    assert board.find_logical_BL_coords(side_i, 0) == (4, 1)
    assert board.find_logical_BL_coords(side_i, 1) == (4, 2)
    assert board.find_logical_BL_coords(side_i, 2) == (4, 3)
    assert board.find_logical_BL_coords(side_i, 3) == (1, 4)
    assert board.find_logical_BL_coords(side_i, 9) == (1, 10)


def test_clear_full_lines():
    ascii = [  # fmt: skip
        "X X",
        " X ",
        "XXX",
    ]

    board = TetrisBoard.from_ascii(ascii)  # Produces a 3x3 board
    sums = np.sum(board.board, axis=1)
    assert np.array_equal(sums, [3, 1, 2])

    clear_full_lines(board.board)
    sums = np.sum(board.board, axis=1)
    assert np.array_equal(sums, [1, 2, 0])


def test_clear_full_lines_not_ones():
    ascii = [  # fmt: skip
        "X X",
        " X ",
        "XXX",
    ]

    fix = TetrisBoard.from_ascii(ascii)  # Produces a 3x3 board
    fix = fix.board

    # The cell value shouldn't matter, as long as it's non-zero
    fix = fix * 2
    fix[0] = np.array([1, 3, -1])

    sums = np.sum(fix, axis=1)
    assert np.array_equal(sums, [3, 2, 4])

    clear_full_lines(fix)
    sums = np.sum(fix, axis=1)
    assert np.array_equal(sums, [2, 4, 0])
    assert np.array_equal(fix[2], np.zeros(3))
