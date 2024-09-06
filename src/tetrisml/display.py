from dataclasses import dataclass
from io import StringIO
import numpy as np
from contextlib import redirect_stdout

from tetrisml.minos import MinoShape


######################
# Rune holding cell
# Note that the emoji are two characters wide
# ▆ 👾 🎉 ◼️ 🌚 ⬛ 🟩
######################


@dataclass
class GridRenderConfig:
    filled: str = "▆ "
    empty: str = "_ "
    last_action: str = "X "
    color: bool = False
    x_labels: bool = False
    y_labels: bool = False


class GridRenderer:
    def __init__(self, config: GridRenderConfig | None = None):
        self.config = config if config is not None else GridRenderConfig()

    def render(
        self,
        grid: np.ndarray,
        last_mino_shape: MinoShape = None,
        lcoords: tuple[int, int] = None,
    ):

        grid = grid.copy()

        if last_mino_shape is not None:

            origin_r = lcoords[0] - 1
            origin_c = lcoords[1] - 1

            mino = last_mino_shape.shape
            for i, row in enumerate(reversed(mino)):
                for j, cell in enumerate(row):
                    if cell == 1:
                        grid[origin_r + i][origin_c + j] = 2

        assert grid.ndim == 2, "Grid must be a 2D array"

        h = len(grid)
        y_gutter = len(str(h)) + 1 if self.config.y_labels else 0

        w = len(grid[0])
        x_gutter = 1

        for i, row in enumerate(reversed(grid)):
            if self.config.y_labels:
                # "NN "
                # " N "
                print(f"{str(len(grid) - i).rjust(y_gutter-1)}", end=" ")

            for cell in row:
                if cell == 1:
                    print(self.config.filled, end="")
                elif cell == 2:
                    print(self.config.last_action, end="")
                else:
                    print(self.config.empty, end="")
            print()

        if self.config.x_labels:
            # [1,2,3...]
            bg = [x for x in range(1, w + 1)]
            # ["1", "2", "3", ... , "9", "0"]
            for i, row in enumerate(bg):
                bg[i] = [x for x in str((row % 10)).ljust(x_gutter)]

            # ["1", " ", "2", " ", "3", ...]
            temp = []
            filler = [" " for x in range(x_gutter)]
            for x in bg:
                temp.append(x)
                temp.append(filler)
            bg = temp
            bg = np.array(bg).transpose()

            pad = " " * (y_gutter + 1) if y_gutter else ""
            # [pad] 1 2 3 4 5 6 7 8 9 0
            print(pad + "".join(bg[0]))

    def __call__(self, *args, **kwargs):
        return self.render(*args, **kwargs)

    @staticmethod
    def render_grid(grid: np.ndarray, config: GridRenderConfig, **kwargs):
        gr = GridRenderer(config)
        return gr.render(grid, **kwargs)

    @staticmethod
    def quick_render(grid, **kwargs):
        config = GridRenderConfig(
            y_labels=True,
        )
        GridRenderer(config).render(grid, **kwargs)

    @staticmethod
    def for_repr(grid: np.ndarray, subject):
        gr = GridRenderer(
            GridRenderConfig(
                x_labels=False,
                y_labels=True,
                color=False,
                filled="🟩",
                empty="⬛",
                last_action="🟨",
            )
        )

        buffer = StringIO()
        with redirect_stdout(buffer):
            gr(grid[:5])
        buffer = buffer.getvalue().strip().split("\n")

        for i, row in enumerate(buffer):
            key = "@" + str(i + 1).rjust(2, "0")
            subject.__setattr__(key, row)


if __name__ == "__main__":

    import sys
    import os

    from tetrisml.board import TetrisBoard

    b = TetrisBoard.from_ascii(
        """
        X
        X X   X
        X X X X
        X X X X
        """,
        h=20,
        w=10,
    )
    mino = MinoShape("T", 0)

    GridRenderer.quick_render(b.board, last_mino_shape=mino, lcoords=(3, 2))

    # c = GridRenderer()
    # c(b.board, mino.shape_id, (3, 0))

    s = dict()
