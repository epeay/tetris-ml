"""
Some tests just to make sure that the numpy stuff is working as expected.
"""

import numpy as np
import pytest
from tetrisml.board import TetrisBoard


@pytest.fixture
def b():
    ascii = [
        "X X",
        " X ",
        "XXX",
    ]

    return TetrisBoard.from_ascii(ascii)


def test_export_board(b: TetrisBoard):
    board_copy = b.export_board()
    assert board_copy.base is None


def test_numpy_memory_sharing(b: TetrisBoard):
    """
    For my own testing and understanding
    """
    b_ref = b.board
    b_copy = b.board.copy()
    b_slice = b.board[2:]

    # Hrmmmm...
    assert np.may_share_memory(b.board, b_ref) == True
    assert b_ref.base is None

    assert np.may_share_memory(b.board, b_copy) == False
    assert b_copy.base is None

    assert np.may_share_memory(b.board, b_slice) == True
    assert b_slice.base is not None

    # Let's change the slice and see if the original board changes
    b_slice[0][0] = 5
    assert b.board[2][0] == 5
    assert b_ref[2][0] == 5
    # The copy is a deep copy, as it doesn't get impacted
    # by the change in the slice
    assert b_copy[2][0] == 1
