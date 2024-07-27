from tetrisml.env import TetrisEnv
from tetrisml import *
import tetrisml


def find_possible_moves(env: TetrisEnv, s: MinoShape) -> list[MinoPlacement]:
    """
    Given a mino and a board, returns a list of possible moves for the mino.
    """

    board: TetrisBoard = env.board

    options = []
    tower_heights = np.array(board.get_tops())

    # This is gonna be inefficient for now

    for c in range(board.width - s.width + 1):
        lcoords = board.find_logical_BL_coords(s, c)

        # Assume placement of Shape:
        # Field:      | Shape:
        # X O O O     |    O O O
        # X X O       |      O
        # X X X       |
        # As shown at coords (2,2)
        #
        # Board height is               [3, 2, 1, 0, ...]
        # Shape height (from bottom) is    [1, 0, 1]

        # [1, 0, 1]
        mino_col_heights = np.array(s.get_bottom_gaps())
        # [2, 1, 0]
        tower_col_heights = tower_heights[c : c + s.width]

        # The gap between the bottom of the mino and the tower
        # [0, 0, 2]
        gaps = mino_col_heights - 1 + lcoords[0] - tower_col_heights

        piece: TetrominoPiece = s.get_piece()
        backup_rows = board.place_shape(s, lcoords)
        reward = tetrisml.calculate_reward(board.board)

        # Revert the board
        for r in range(len(backup_rows)):
            board.board[lcoords[0] - 1 + r] = backup_rows[r]

        placement = MinoPlacement(s, lcoords, gaps.tolist(), reward)
        options.append(placement)

    return options


def get_future_reward(e: TetrisEnv, s: MinoShape, lcoords: tuple[int, int]) -> float:
    """
    Given a board, mino, and placement coords, returns the reward of that
    placement. Temporarily modifies the board, but reverts.
    """
    backup_rows = e.board.place_shape(s, lcoords)
    reward = e._calculate_reward()

    # Revert the board
    for r in range(len(backup_rows)):
        e.board.board[lcoords[0] - 1 + r] = backup_rows[r]

    return reward
