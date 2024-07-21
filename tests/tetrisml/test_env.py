import pytest
import sys

from tetrisml.env import MinoBag
import sys
 
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
