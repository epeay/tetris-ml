from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt


def get_color(pack):
    # Define color scale
    if pack == 1.0:
        return (255, 215, 0)  # Golden color for completed row
    else:
        # Use a color map from matplotlib for blue shades
        cmap = plt.get_cmap("Blues")
        rgba = cmap(pack)
        return tuple(int(255 * x) for x in rgba[:3])  # Convert to RGB


def create_board_image(board_states, scale=10):
    num_states = len(board_states)
    height, width = len(board_states[0]), len(board_states[0][0])

    # Create an image with one column per board state and height as the board's height
    image = Image.new("RGB", (num_states * scale, height * scale))
    draw = ImageDraw.Draw(image)

    for col, board in enumerate(board_states):
        for row in range(height):
            pack = sum(board[row]) / width
            color = get_color(pack)
            # Draw the scaled block
            x1, y1 = col * scale, (height - row - 1) * scale
            x2, y2 = x1 + scale - 1, y1 + scale - 1
            draw.rectangle([x1, y1, x2, y2], fill=color)

    return image


# Example board states (list of 2D arrays)
board_states = [
    np.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        ]
    ),
    np.array(
        [
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
        ]
    ),
    # Add more board states as needed
]

# Create and save the image
image = create_board_image(board_states, scale=10)
image.save("board_states_scaled.png")
