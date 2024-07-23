import gymnasium as gym
import numpy as np

from numpy import ndarray as NDArray


class Tetrominos:
    O = 0
    I = 1
    S = 2
    Z = 3
    T = 4
    J = 5
    L = 6
    DOT = 7
    USCORE = 8

    base_patterns = {
        # X X
        # X X
        O: np.array([[1, 1], [1, 1]]),
        # X X X X
        I: np.array([[1, 1, 1, 1]]),
        # _ X X
        # X X _
        S: np.array([[0, 1, 1], [1, 1, 0]]),
        Z: np.array([[1, 1, 0], [0, 1, 1]]),
        T: np.array([[1, 1, 1], [0, 1, 0]]),
        J: np.array([[1, 0, 0], [1, 1, 1]]),
        L: np.array([[0, 0, 1], [1, 1, 1]]),
        DOT: np.array([[1]]),
        USCORE: np.array([[1, 1]]),
    }

    int_name_lookup = [
        (O, "O"),
        (I, "I"),
        (S, "S"),
        (Z, "Z"),
        (T, "T"),
        (J, "J"),
        (L, "L"),
        (DOT, "D"),
        (USCORE, "U"),
    ]

    # Stores patterns for each tetromino, at each rotation
    cache = {}

    def get_num_tetrominos():
        return len(Tetrominos.base_patterns.keys())

    @staticmethod
    def shape_name(shape_id: int):

        # Find the shape name
        for id, name in Tetrominos.int_name_lookup:
            if id == shape_id:
                return name

        raise ValueError(f"Invalid shape id {shape_id}")

    @staticmethod
    def make_by_char(c: str):
        for id, char in Tetrominos.int_name_lookup:
            if char == c:
                return Tetrominos.make(id)

        raise ValueError(f"Invalid shape character {c}")

    @staticmethod
    def get_id_by_char(c: str):
        for id, char in Tetrominos.int_name_lookup:
            if char == c:
                return id

        raise ValueError(f"Invalid shape character {c}")

    @staticmethod
    def make(shape: int, rot=0):
        """
        shape:
        """
        if not Tetrominos.cache:
            for id, pattern in Tetrominos.base_patterns.items():
                Tetrominos.cache[id] = [
                    np.array(pattern),
                    np.array(np.rot90(pattern)),
                    np.array(np.rot90(pattern, 2)),
                    np.array(np.rot90(pattern, 3)),
                ]

        if shape not in Tetrominos.base_patterns.keys():
            raise ValueError("Invalid shape")

        ret = TetrominoPiece(shape, Tetrominos.cache[shape])
        for _ in range(rot):
            ret.rotate()
        return ret


class TetrominoPiece:

    BLOCK = "▆"

    def __init__(self, shape: int, patterns):
        self.shape: int = shape
        self.pattern_list: list[NDArray] = patterns
        self.pattern: NDArray = patterns[0]
        self.rot = 0

    def __str__(self) -> str:
        return f"TetrominoPiece(shape={Tetrominos.shape_name(self.shape)}, rot={self.rot*90}, pattern= {self.printable_pattern(oneline=True)})"

    def printable_pattern(self, oneline=False):
        ret = []
        pattern = self.get_pattern()
        for i, row in enumerate(pattern):
            row_str = " ".join([str(c) for c in row])
            ret.append(row_str)

            if not oneline:
                ret.append("\n")
            else:
                if i < len(pattern) - 1:
                    ret.append(
                        " / ",
                    )
        ret = "".join(ret).replace("1", TetrominoPiece.BLOCK).replace("0", "_")
        return "".join(ret)

    def to_dict(self):
        return {
            "shape": self.shape,
            "pattern": self.pattern.tolist(),
        }

    def get_pattern(self):
        return self.pattern

    def get_shape(self, rot):
        """
        Returns the pattern for the specified rotation.

        A 'Z' mino would return [[1,1,0],[0,1,1]] for rot=0
        XX_
        _XX
        """
        return self.pattern_list[rot]

    def rotate(self):
        """Rotates IN PLACE, and returns the new pattern"""
        self.rot = (self.rot + 1) % 4
        self.pattern = self.pattern_list[self.rot]
        return self.pattern

    def get_height(self, rot=None):
        if rot:
            return len(self.get_shape(rot))
        else:
            return len(self.get_pattern())

    def get_width(self, rot=None):
        if rot:
            return len(self.get_shape(rot)[0])
        else:
            return max([len(x) for x in self.get_pattern()])

    def get_bottom_offsets(self):
        """
        For each column in the shape, returns the gap between the bottom of
        the shape (across all columns) and the bottom of the shape in that
        column.

        Returned values in the list would expect to contain at least one 0, and
        no values higher than the height of the shape.

        For example, an S piece:
        _ X X
        X X _

        Would have offsets [0, 0, 1] in this current rotation. This method is
        used in determining if a piece will fit at a certain position
        in the board.
        """
        pattern = self.get_pattern()
        # pdb.set_trace()
        ret = [len(pattern) + 1 for x in range(len(pattern[0]))]
        # Iterates rows from top, down
        for ri in range(len(pattern)):
            # Given a T shape:
            # X X X
            # _ X _
            # Start with row [X X X] (ri=0, offset=1)
            row = pattern[ri]
            # print(f"Testing row {row} at index {ri}")
            for ci, col in enumerate(row):
                if col == 1:
                    offset = len(pattern) - ri - 1
                    ret[ci] = offset

            # Will return [1, 0, 1] for a T shape

        if max(ret) >= len(pattern):
            print(f"Pattern:")
            print(pattern)
            print(f"Bottom Offsets: {ret}")
            print(f"Shape: {self.shape}")
            raise ValueError("Tetromino pattern has incomplete bottom offsets")

        return ret

    def get_top_offsets(self):
        """
        Returns the height of the shape at each column.

        For example, an S piece:
        _ X X
        X X _

        Would have offsets [1, 2, 2] in this current rotation. This provides
        guidance on how to update the headroom list.

        Ideally we should cache this.
        """
        pattern = self.get_pattern()
        ret = [0 for x in len(pattern[0])]
        for ri, row in enumerate(
            range(
                pattern,
            )
        ):
            for col in pattern[row]:
                if pattern[row][col] == 1:
                    ret[col] = max(ret[col], row)
        return ret


Tetrominos.std_bag = [
    Tetrominos.O,
    Tetrominos.I,
    Tetrominos.S,
    Tetrominos.Z,
    Tetrominos.T,
    Tetrominos.J,
    Tetrominos.L,
]
