from tetrisml.board import *
from tetrisml.tetrominos import *


def test_from_ascii():
    ascii = ["X X", 
             " X ", 
             "XXX"]

    board = TetrisBoard.from_ascii(ascii)
    expected = np.array([[1,1,1], [0,1,0], [1,0,1]])
    assert np.array_equal(board.board, expected)

    # Test padding
    board = TetrisBoard.from_ascii(ascii, h=20, w=10)
    expected = np.zeros((20, 10), dtype=int)
    expected[0:3,0:3] = np.array([[1,1,1], [0,1,0], [1,0,1]])
    assert np.array_equal(board.board, expected)




def test_find_logical_BL_coords():
    ascii = ["X X", 
             " X ", 
             "XXX"]
    
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
