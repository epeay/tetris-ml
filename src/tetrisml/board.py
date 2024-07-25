import numpy as np
import os

from numpy import ndarray as NDArray

from .minos import MinoShape
from .tetrominos import TetrominoPiece


class TetrisBoard:

    BLOCK = "▆"

    def __init__(self, matrix: NDArray, height):
        self.play_height = height
        self.height = len(matrix)
        self.width = len(matrix[0])
        self.board: NDArray = np.array(matrix)

    def reset(self):
        self.board.fill(0)
        self.piece = None

    def remove_tetris(self):
        to_delete = []
        for r, row in enumerate(self.board):
            if sum(row) == self.width:
                to_delete.append(r)

        if to_delete:

            # TODO Handle this more efficiently
            # I believe BOTH of these operations make copies of the source data
            self.board = np.delete(self.board, to_delete, axis=0)

            # Odd workaround because vscode is messing with the memory management
            # of numpy by holding a reference to the board, which prevents the
            # resize in place.
            refcheck = True  # This is the default
            if "ISDEBUG" in os.environ.keys():
                refcheck = False

            self.board.resize((self.height, self.width), refcheck=refcheck)

        return len(to_delete)

    def export_board(self):
        return self.board.copy()

    def place_shape(
        self, s: MinoShape, logical_coords: tuple[int, int]
    ) -> list[NDArray]:
        piece = s.get_piece()
        return self.place_piece(piece, logical_coords)

    def place_mino(
        self, mino: TetrominoPiece, rot: int, logical_coords: tuple[int, int]
    ) -> list[NDArray]:
        """
        A simple wrapper around place_piece, knowing that I want to make TetrominoPiece
        stateless.
        """
        old_rot = mino.rot
        mino.rot = rot
        ret = self.place_piece(mino, logical_coords)
        mino.rot = old_rot
        return ret

    def place_piece(
        self, piece: TetrominoPiece, logical_coords, alt_board=None
    ) -> list[NDArray]:
        """
        Places a piece at the specified column. Dynamically calculates correct
        height for the piece.

        piece: a TetrominoPiece object
        logical_coords: The logical row and column for the bottom left
            of the piece's pattern
        alt_board: An optional board to place the piece on. If not provided,
            the board attribute is used.
        """
        pattern = piece.get_pattern()
        board = self.board if alt_board is None else alt_board

        lrow = logical_coords[0]
        lcol = logical_coords[1]

        lr = logical_coords[0]
        row_backups = [x.copy() for x in board[lr - 1 : lr - 1 + piece.get_height()]]

        p_height = piece.get_height()

        for r in range(p_height):
            pattern_row = pattern[len(pattern) - 1 - r]
            board_row = board[lrow - 1 + r]

            for i, c in enumerate(pattern_row):
                # Iff c is 1, push it to the board
                board_row[lcol - 1 + i] |= c

        return row_backups

    def find_logical_BL_coords(self, mino: MinoShape, col):
        """
        Assumes the piece fits on the board, horizontally. The piece WILL fit
        vertically, as there are 4 empty rows at the top of the board, which if
        utilized, trigger game over.

        Returns the logical row and column of the bottom left corner of the
        pattern, such that when placed, the piece will sit flush against existing
        tower parts, and not exceed the max board height.

        Given:
        BOARD       PIECE
        5 _ _ _ _
        4 _ _ _ X
        3 _ _ X X   X X X X
        2 _ X X _
        1 X X X X

        Returns (5, 1)

        Given:
        BOARD       PIECE    COL
        5 _ _ _ _
        4 _ _ _ X
        3 _ _ X X   X X X    1 (lcol 2)
        2 _ X X _     X
        1 X X X X

        Returns (3, 1)

        piece: a TetrominoPiece object
        col: zero-index column to place the 0th column of the piece.
        """

        bottom_offsets = np.array(mino.get_piece().get_bottom_offsets())
        board_heights = np.array(self.get_tops()[col : col + mino.width])

        p_height = mino.height
        p_width = mino.width
        can_place = False

        for place_row in range(min(board_heights) + 1, max(board_heights) + 2):
            bottom_clears_board = all((bottom_offsets + place_row) > board_heights)
            if bottom_clears_board:
                break

        return (place_row, col + 1)

    @staticmethod
    def from_ascii(ascii: list[str], h: int = None, w: int = None):
        """
        Create a TetrisBoard object from an ASCII representation of a board.
        This is an important method for bootstrapping tests.
        """

        ascii = ascii[::-1]
        space_chars = ("_", " ", ".", "0")

        ah = len(ascii)
        aw = max([len(x) for x in ascii])

        if h is not None and h < ah:
            raise ValueError(
                "Requested height of board is less than ASCII representation"
            )

        if w is not None and w < aw:
            raise ValueError(
                "Requested width of board is less than ASCII representation"
            )

        h = ah if h is None else max(h, ah)
        w = aw if w is None else max(w, aw)

        ret = np.zeros((h, w), dtype=int)

        for ri, row in enumerate(ascii):
            for ci, col in enumerate(row):
                if col not in space_chars:
                    ret[ri][ci] = 1

        return TetrisBoard(ret, h)

    def find_logical_BL_placement(self, piece: TetrominoPiece, col):

        return self.find_logical_BL_coords(MinoShape(piece.shape, 0), col)

    @staticmethod
    def render_state(
        board,
        highlight_shape: MinoShape = None,
        highlight_bl_coords=None,
        color=True,
        title="",
    ):
        board = board.copy()
        output = False

        if highlight_shape is not None and highlight_bl_coords is not None:
            shape = highlight_shape.shape
            p_height = len(shape)
            lrow = highlight_bl_coords[0]
            lcol = highlight_bl_coords[1]

            for r in range(p_height):
                pattern_row = shape[len(shape) - 1 - r]
                board_row = board[lrow - 1 + r]

                for i, c in enumerate(pattern_row):
                    # Iff c is 1, push it to the board
                    if c == 1:
                        board_row[lcol - 1 + i] = 2

        if title is not None:
            print(f"== {title} ==")

        for i, row in enumerate(reversed(board)):
            if sum(row) == 0 and not output:
                continue
            else:
                output = True

            for cell in row:
                if cell == 2:
                    print(f"\033[36m{TetrisBoard.BLOCK}\033[0m", end=" ")
                elif cell == 1:
                    print(TetrisBoard.BLOCK, end=" ")
                else:
                    print("_", end=" ")
            print()

        print("=======================================")

    def render(self):
        output = False
        for i, row in enumerate(reversed(self.board)):
            # if sum(row) == 0 and not output:
            #     continue
            # else:
            #     output = True

            output = True

            print(f"{(self.height -i) % 10} ", end="")
            for cell in row:
                if cell == 1:
                    print(TetrisBoard.BLOCK, end=" ")
                else:
                    empty = "_"
                    if (self.height - i) > 20:
                        empty = "X"

                    print(empty, end=" ")
            print()

        if not output:
            print("<<EMPTY BOARD>>")

    def get_tops(self):
        """
        Gets the height of each column on the board.
        This is gonna be inefficient for now.

        A board with only an I at the left side would return [4, 0, 0, ...]
        """
        tops = [0 for _ in range(self.width)]
        for r, row in enumerate(self.board):
            if sum(row) == 0:
                break

            for col, val in enumerate(row):
                if val == 1:
                    tops[col] = r + 1

        return tops
