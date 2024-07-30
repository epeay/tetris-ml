import matplotlib.pyplot as plt
import torch
import numpy as np

from model import TetrisCNN
from numpy import ndarray as NDArray


# Function to visualize feature maps
def visualize_board_and_feature_maps(
    board: NDArray, feature_maps, layer_name, img_name
):
    plt.ioff()  # Turn off interactive mode

    num_maps = feature_maps.shape[1]
    rows = 4
    cols = int(np.ceil((num_maps + 1) / rows))
    fig, axs = plt.subplots(rows, cols, figsize=(20, 10))

    axs = axs.flatten()

    # Display the board state
    board_np = np.flipud(board[0, 0])
    axs[0].imshow(board_np, cmap="Blues")
    axs[0].axis("off")
    axs[0].set_title("Board")

    # Display the feature maps
    for i in range(num_maps):
        flipped_map = np.flipud(feature_maps[0, i])
        axs[i + 1].imshow(flipped_map, cmap="gray")
        axs[i + 1].axis("off")
        axs[i + 1].set_title(f"FM{i + 1}")

    plt.savefig(img_name)
    plt.close()
    plt.ion()  # Turn interactive mode back on


def activations_heatmap(
    board, heatmap, title="Grad-CAM Heatmap", img_name="heatmap.png"
):
    plt.ioff()
    board = board[0, 0]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display the board
    axs[0].imshow(board, cmap="Blues")
    axs[0].set_title("Board State")
    axs[0].invert_yaxis()  # Invert the y-axis
    num_rows, num_cols = board.shape
    axs[0].set_xticks(np.arange(num_cols))
    axs[0].set_yticks(np.arange(num_rows))
    axs[0].set_xticklabels(np.arange(num_cols))
    axs[0].set_yticklabels(np.arange(num_rows))

    # Display the heatmap
    im = axs[1].imshow(heatmap, cmap="viridis")
    axs[1].set_title(title)
    axs[1].invert_yaxis()  # Invert the y-axis
    num_rows, num_cols = heatmap.shape
    axs[1].set_xticks(np.arange(num_cols))
    axs[1].set_yticks(np.arange(num_rows))
    axs[1].set_xticklabels(np.arange(num_cols))
    axs[1].set_yticklabels(np.arange(num_rows))

    # Add color bar
    fig.colorbar(im, ax=axs[1])

    plt.savefig(img_name)
    plt.close()
    plt.ion()


# Function to apply Grad-CAM and save heatmap
def grad_cam(model, input, linear_data, target_layer: torch.nn.modules.conv.Conv2d):
    model.eval()

    tl_activations = []

    def hook_fn(module, input, output):
        tl_activations.append(output)

    handle = target_layer.register_forward_hook(hook_fn)
    model_out = model(input, linear_data)
    handle.remove()

    tl_out = tl_activations[0]

    # output = otput[0]  # Assuming single output for simplicity
    model_choice = model_out.max()
    model.zero_grad()
    model_choice.backward(retain_graph=True)

    gradients = target_layer.weight.grad
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    activations = tl_out.detach()

    for i in range(len(pooled_gradients)):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    return heatmap.numpy()


def visualize(m: TetrisCNN, d4_board: NDArray, step_count):
    pass
    # Save feature maps from conv1 and conv2 layers

    # # Use Grad-CAM
    # heatmap = grad_cam(m, d4_board, m.conv2)
    # plt.imshow(heatmap, cmap='viridis')
    # plt.colorbar()
    # plt.savefig(f"grad_cam_heatmap-s{step_count}.png")
    # plt.close()
