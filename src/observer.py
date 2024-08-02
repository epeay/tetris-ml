import wandb
from tetrisml.base import BasePlayer
from tetrisml.env import PlaySession
from tetrisml.dig import DigEnv
from tetrisml.tetrominos import Tetrominos
from tetrisml.minos import MinoShape
from tetrisml.base import CallbackHandler, ActionContext
from collections import defaultdict


class ActionTracker:

    T_GUESS = "guess"
    T_PREDICT = "predict"

    def __init__(self):
        # Initialize the nested dictionary to track actions
        self.actions = defaultdict(lambda: defaultdict(int))

    def register_action(self, action_type: str, is_correct: bool):
        # Convert boolean accuracy to string ('1' for True, '0' for False)
        accuracy_str = "1" if is_correct else "0"
        # Increment the corresponding counter in the nested dictionary
        self.actions[action_type][accuracy_str] += 1
        self.actions[action_type]["total"] += 1
        self.actions[action_type]["accuracy"] = (
            self.actions[action_type]["1"] / self.actions[action_type]["total"]
        )

    def emit(self, episode: int):
        # Log the actions to Weights & Biases
        self.wandb_log(episode)

    def get_debug_output(self):
        # Predict: X/Y (Z%)
        guess = f"{self.actions[self.T_GUESS]['1']}/{self.actions[self.T_GUESS]['total']} ({self.actions[self.T_GUESS]['accuracy']:.2%})"
        predict = f"{self.actions[self.T_PREDICT]['1']}/{self.actions[self.T_PREDICT]['total']} ({self.actions[self.T_PREDICT]['accuracy']:.2%})"
        return {"Predict": predict, "Guess": guess}

    def wandb_log(self, episode: int):
        log = {
            "episode": episode,
            "actions": self.to_dict(),
        }
        wandb.log(log)
