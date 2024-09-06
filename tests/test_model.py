import numpy as np
from numpy import dtype
from model import ModelState
import pytest


@pytest.fixture
def state() -> ModelState:
    grid = np.zeros((20, 10), dtype=int)
    fix = ModelState(grid)
    fix.set_upcoming_minos([0, 1, 2, 3])
    return fix


def test_ModelState_set_upcoming_minos(state: ModelState):
    state.set_upcoming_minos([])
    assert len(state.get_upcoming_minos()) == 0

    ex = [0, 1, 2, 3]
    state.set_upcoming_minos(ex)
    assert len(state.get_upcoming_minos()) == 4
    assert state.get_upcoming_minos() == ex


def test_ModelState_import_one_hots(state: ModelState):

    state.set_upcoming_minos([])
    #              [       ][       ][       ][       ]
    ohs = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0])  # Extra zero
    with pytest.raises(ValueError):
        state.import_one_hots(ohs, 3)
    assert state.get_upcoming_minos() == []

    ohs = ohs[:-1]  # len 12

    with pytest.raises(ValueError):
        state.import_one_hots(ohs, 12)  # invalid one-hot
    assert state.get_upcoming_minos() == []

    with pytest.raises(ValueError):
        state.import_one_hots(ohs, 2)  # Data after empty data
    assert state.get_upcoming_minos() == []

    state.set_upcoming_minos([])
    state.import_one_hots(ohs, 3)
    assert state.get_upcoming_minos() == [0, 1, 2, 0]


def test_ModelState_get_one_hots(state: ModelState):

    state.set_upcoming_minos([0, 1, 2])
    with pytest.raises(ValueError):
        state.get_one_hot_data(2)  # Invalid width

    state.set_upcoming_minos([])
    with pytest.raises(ValueError):
        state.get_one_hot_data(2)  # Empty queue

    expect = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])  # 0, 1, 2
    state.set_upcoming_minos([0, 1, 2])
    actual = state.get_one_hot_data(3)
    assert np.array_equal(expect, actual)


def test_ModelState_to_dict(state: ModelState):
    state.set_upcoming_minos([0, 1, 2, 3])

    d = state.to_dict()
    assert d["upcoming"] == [0, 1, 2, 3]
    assert d["board"] == np.zeros((20, 10), dtype=int).tolist()
