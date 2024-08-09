from .tetrominos import TetrominoPiece, Tetrominos


class MinoShape:
    """
    A MinoShape is a single rotation of a Tetromino piece. It is a 2D array.
    Importantly, this class is meant to be immutable.
    """

    def __init__(self, shape_id: int | str, rot: int = 0):

        if isinstance(shape_id, str) and len(shape_id) == 1:
            shape_num = Tetrominos.get_id_by_char(shape_id)
            shape_letter = shape_id
        elif isinstance(shape_id, int):
            shape_num = shape_id
            shape_letter = Tetrominos.shape_name(shape_id)
        else:
            raise ValueError(f"Invalid shape_id '{shape_id}'")

        self.shape: list[list[int]] = Tetrominos.make(shape_num, rot).pattern
        self.shape_id: int = shape_num
        self.shape_rot: int = rot
        self.letter: str = shape_letter
        self._bottom_gaps: list[int] = None

    @property
    def height(self):
        return len(self.shape)

    @property
    def width(self):
        return len(self.shape[0])

    def __str__(self):
        return f"MinoShape(shape={Tetrominos.shape_name(self.shape_id)}, rot={self.shape_rot}, pattern=[{self.printable_pattern(oneline=True)}])"

    def __repr__(self):
        return self.__str__()

    def printable_pattern(self, oneline=False):
        ret = []
        pattern = self.shape
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

    def to_jsonable(self):
        return {
            "id": self.shape_id,
            "name": Tetrominos.shape_name(self.shape_id),
            "rot": self.shape_rot,
            "shape": self.shape.tolist(),
        }

    @staticmethod
    def from_jsonable(data: dict):
        return MinoShape(data["id"], data["rot"])

    def by_name(self):
        return Tetrominos.shape_name(self.shape_id)

    def get_piece(self) -> TetrominoPiece:
        """
        Backtrack to the TetrominoPiece of this shape.
        """
        ret = Tetrominos.make(self.shape_id, self.shape_rot)
        return ret

    def get_bottom_gaps(self):
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

        if self._bottom_gaps:
            return self._bottom_gaps

        pattern = self.shape
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

        self._bottom_gaps = ret

        return self._bottom_gaps


class MinoPlacement:
    def __init__(
        self,
        shape: MinoShape,
        bl_coords: tuple[int, int],
        gaps_by_col: list[int],
        reward: float,
    ) -> None:
        self.shape = shape
        self.bl_coords = bl_coords
        self.reward: float = reward

        # At which columns does the piece not sit flush?
        # Field:      | Shape:
        # X O O O     |    O O O
        # X X O       |      O
        # X X X       |
        # The gaps are [0, 0, 2]
        self.gaps: list[int] = gaps_by_col
        self.empty_tiles_created = sum(self.gaps)
        self.is_flush = self.empty_tiles_created == 0

    def to_jsonable(self):
        return {
            "shape": self.shape.to_jsonable(),
            "bl_coords": self.bl_coords,
            "gaps": self.gaps,
            "reward": self.reward,
            "empty_tiles_created": self.empty_tiles_created,
            "is_flush": self.is_flush,
        }

    @staticmethod
    def from_jsonable(data: dict):
        shape = MinoShape.from_jsonable(data["shape"])
        return MinoPlacement(shape, data["bl_coords"], data["gaps"], data["reward"])
