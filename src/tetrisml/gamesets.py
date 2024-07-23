import json
import os
import pandas as pd
import IPython
import matplotlib.pyplot as plt
import tensorflow

plt.ion()

from .logging import GameHistory
from utils import ipython


class GameRuns:
    def __init__(self, storage_root: str):
        self.runs = []
        self.storage_root: str = storage_root
        self.sets_loaded = {}
        self.games_by_id = {}
        self.games_by_run = {}

    def load(self, id: str):

        if id in self.sets_loaded:
            return

        filename = f"{id}.json"
        if not os.path.exists(os.path.join(self.storage_root, filename)):
            raise FileNotFoundError(f"Game set {id} not found")

        set_data = json.load(open(os.path.join(self.storage_root, filename), "r"))

        games = [GameHistory.from_jsonable(x[1]) for x in set_data["games"].items()]

        for g in games:
            self.games_by_id[g.id] = g

        ret = sorted(games, key=lambda x: x.unix_ts)
        self.games_by_run[id] = ret
        return ret

    def reward_plot(self, game_id: str):
        game: GameHistory = self.games_by_id[game_id]
        ds = pd.Series([x.reward for x in game.placements])
        ds.plot()

    def cumulative_reward_plot(self, games: list[GameHistory]):
        df = pd.DataFrame()
        plt.ion()

        # Cumulative reward over the duration of the game
        series = dict(
            [
                (g.id, pd.Series([x.reward for x in g.placements]).cumsum())
                for g in games
            ]
        )
        df = pd.DataFrame(series)

        # For coloring purposes, we're counting on the dict to preserve
        # the game odering across its keys. This should be the case in
        # Python 3.7+

        # Create a colormap
        cmap = plt.get_cmap("Blues")
        colors = [cmap(i / len(df.columns)) for i in range(len(df.columns))]

        # Plot all columns
        plt.figure(figsize=(14, 7))
        for i, column in enumerate(df):
            plt.plot(df[column], label=column, alpha=0.6, color=colors[i])

        plt.title("Cumulative Rewards of Different Games")
        plt.xlabel("Step")
        plt.ylabel("Cumulative Reward")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=2, fontsize="small")
        plt.grid(True)
        plt.show(block=True)

    def visualize_feature_maps(ax, model, layer_name, input_image):
        from tensorflow.keras import Model

        layer_model = Model(
            inputs=model.input, outputs=model.get_layer(layer_name).output
        )
        feature_maps = layer_model.predict(np.expand_dims(input_image, axis=0))
        num_feature_maps = feature_maps.shape[-1]
        for i in range(num_feature_maps):
            if i < len(ax):
                ax[i].imshow(feature_maps[0, :, :, i], cmap="viridis")
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                ax[i].set_title(f"Feature Map {i+1}")

    def visualize_filters(ax, model, layer_name):
        filters, biases = model.get_layer(layer_name).get_weights()
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        num_filters = filters.shape[-1]
        for i in range(num_filters):
            if i < len(ax):
                ax[i].imshow(filters[:, :, :, i], cmap="viridis")
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                ax[i].set_title(f"Filter {i+1}")
