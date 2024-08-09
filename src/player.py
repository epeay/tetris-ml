import json
import random
from tetrisml import MinoShape, TetrisEnv, BasePlayer
from tetrisml.tetris_logging import GameHistory
from tetrisml.env import ModelAction
from tetrisml.minos import MinoPlacement
from cheating import find_possible_moves
import numpy as np


class CheatingPlayer(BasePlayer):
    """
    A player that makes the best choice based on running the reward function
    on all possible moves. Though it only looks one move in the future.
    """

    def __init__(self, seed=None):
        super().__init__()
        self.rng = random.Random(seed)
        pass

    def play(self, e: TetrisEnv):
        placement = self._find_best_move(e)
        return ModelAction((placement.bl_coords[1] - 1, placement.shape.shape_rot))

    def _find_best_move(self, e: TetrisEnv):
        possibilities = []
        b = e.board
        mino = e.current_mino
        best_reward = -np.inf
        # TODO: Some minos don't need four rotations
        for i in range(4):
            possibilities.append(
                np.array(find_possible_moves(e, MinoShape(mino.shape_id, i)))
            )

        possibilities: list[MinoPlacement] = np.concatenate(possibilities)
        highest_reward_choices = []

        for p in possibilities:
            if p.reward == best_reward:
                highest_reward_choices.append(p)
            if p.reward > best_reward:
                best_reward = p.reward
                highest_reward_choices = [p]

        self.rng.shuffle(highest_reward_choices)

        # Further filter by gaps
        highest_reward_choices = sorted(
            highest_reward_choices, key=lambda x: sum(x.gaps), reverse=False
        )

        return highest_reward_choices[0]


class RandomPlayer(BasePlayer):
    """
    Plays from random moves. Complete guesses.
    """

    def __init__(self, seed=None):
        super().__init__()
        self.rng = random.Random(seed)
        pass

    def play(self, e: TetrisEnv):
        rot = self.rng.randint(0, 3)  # Range [0, 3]
        m = MinoShape(e.current_mino.shape_id, rot)
        # if mino is 2 columns, on a 10 col board, placement on col 9 is fine,
        # but 10 would go beyond the board.
        max_col = e.board.width - m.width + 1
        col = self.rng.randint(1, max_col)  # Range [1, max_col]
        return ModelAction((col - 1, rot))


class PlaybackPlayer(BasePlayer):
    """
    Meant to recreate games that have been played before. Useful for
    benchmarking and debugging by keeping gameplay consistent.
    """

    def __init__(self, actions: list[ModelAction]):
        super().__init__()
        self.actions = actions
        self.action_index = 0
        self.playitr = iter(self.actions)

    def play(self, e: TetrisEnv):
        # Throws StopIteration if the list is exhausted
        next_play = next(self.playitr)

        if next_play.shape.shape_id != e.current_mino.shape_id:
            print("Mismatched shapes")
            print(
                f"Expected {e.current_mino.shape_id} but got {next_play.shape.shape_id}"
            )
            print(f"Planned move: {next_play.shape}")
            print(f"Current piece: {e.current_mino}")
            raise ValueError("Mismatched shapes")

        return (next_play.bl_coords[1] - 1, next_play.shape.shape_rot)

    @staticmethod
    def from_file(path: str):
        with open(path, "r") as f:
            playback = json.load(f)
            playback = [
                GameHistory.from_jsonable(g) for g in playback["games"].values()
            ]

        actions = []

        for game in playback:
            for p in game.placements:
                actions.append(ModelAction(p.bl_coords[1] - 1, p.shape.shape_rot))

        return PlaybackPlayer(actions)

    def load(self, actions: list[ModelAction]):
        self.actions = actions
        self.playitr = iter(self.actions)
