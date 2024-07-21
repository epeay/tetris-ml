from tetrisml.minos import *
import numpy as np



def test_MinoShape():
    side_i = MinoShape(Tetrominos.I, 0)
    assert side_i.shape_rot == 0
    assert side_i.height == 1
    assert side_i.width == 4

    expected_shape = np.ones((1,4))

    assert side_i.shape.shape == expected_shape.shape



