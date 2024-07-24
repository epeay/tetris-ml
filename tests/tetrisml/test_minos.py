from tetrisml.minos import *
import numpy as np


def test_MinoShape():
    side_i = MinoShape(Tetrominos.I, 0)
    assert side_i.shape_rot == 0
    assert side_i.height == 1
    assert side_i.width == 4

    expected_shape = np.ones((1, 4))

    assert side_i.shape.shape == expected_shape.shape


def test_clockwise_rotations():
    """
    Assert clockwise rotations of the J piece
    """
    # fmt: off
    expect = [
        np.array([[1, 0, 0], 
                  [1, 1, 1]]),

        np.array([[1, 1],
                  [1, 0], 
                  [1, 0]]),

        np.array([[1, 1, 1],
                  [0, 0, 1]]),

        np.array([[0, 1],
                  [0, 1], 
                  [1, 1]])
    ]
    # fmt: on

    for i in range(4):
        j = MinoShape(Tetrominos.J, i)
        assert (j.shape == expect[i]).all()
