import random
import subprocess


def get_git_hash():

    # TODO What if we're not in a git repo?

    ret = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    ret = ret.decode("ascii").strip()
    return ret


# fmt: off
adjectives = [
    "angry", "basic", "brave", "brief", "calm", "chill", "clean", "clear",
    "close", "crazy", "crisp", "cruel", "dense", "dirty", "dizzy", "eager",
    "early", "empty", "faint", "false", "fancy", "fatal", "final", "flash",
    "fresh", "funny", "giant", "grand", "green", "happy", "harsh", "heavy",
    "human", "ideal", "jolly", "large", "later", "legal", "light", "local",
    "lucky", "magic", "major", "merry", "messy", "minor", "moist", "moral",
    "mushy", "nasty", "neat", "noble", "noisy", "odd", "oily", "pale",
    "plain", "plump", "proud", "quick", "quiet", "rapid", "ready", "right",
    "rough", "round", "salty", "scary", "sharp", "shiny", "silly", "sleek",
    "small", "smart", "solid", "sound", "spicy", "stark", "stiff", "still",
    "storm", "stout", "sweet", "tasty", "tough", "vivid", "young"
]

common_nouns = [
    "apple", "bread", "chair", "clock", "cloud", "dress", "drink", "field",
    "flame", "glass", "grape", "heart", "horse", "house", "light", "money",
    "night", "piano", "plane", "plant", "river", "shape", "shirt", "sleep",
    "smile", "sound", "stone", "table", "thumb", "train", "voice", "water",
    "apple", "bread", "chair", "clock", "dress", "drink", "field", "flame",
    "glass", "grape", "heart", "horse", "house", "light", "money", "night",
    "piano", "plane", "plant", "river", "shape", "shirt", "sleep", "smile",
    "sound", "stone", "table", "thumb", "train", "voice", "water", "world",
    "apple", "bread", "chair", "clock", "cloud", "dress", "drink", "field",
    "flame", "glass", "grape", "heart", "horse", "house", "light", "money",
    "night", "piano", "plane", "plant", "river", "shape", "shirt", "sleep",
    "smile", "sound", "stone", "table", "thumb", "train", "voice", "water"
]
# fmt: on


def word_id():
    """
    word-based id
    """
    return f"{random.choice(adjectives)}{random.choice(common_nouns)}"


def ipython():
    """
    Start an IPython session
    """
    import IPython

    IPython.embed()
