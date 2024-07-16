import datetime
import time
import random

from datetime import datetime

from .minos import MinoPlacement

class TetrisGameRecord:
    def __init__(self):
        self.id = None  # Populated later
        self.moves = 0
        self.invalid_moves = 0
        self.lines_cleared = 0
        self.cleared_by_size = {
            1: 0,
            2: 0,
            3: 0,
            4: 0
        }
        self.boards = []
        self.pieces = []
        self.placements = []
        self.rewards = []
        self.outcomes = []
        self.cumulative_reward = 0
        self.is_predict = []
        self.episode_start_time = time.monotonic_ns()
        self.episode_end_time = None
        self.duration_ns = None
        self.agent_info = {}

    def to_jsonable(self):
        # omitting boards
        ret = {
            "id": self.id,
            "moves": self.moves,
            "invalid_moves": self.invalid_moves,
            "lines_cleared": self.lines_cleared,
            "cleared_by_size": self.cleared_by_size,
            "pieces": self.pieces,
            "placements": self.placements,
            "rewards": self.rewards,
            "outcomes": self.outcomes,
            "cumulative_reward": self.cumulative_reward,
            "episode_start_time": self.episode_start_time,
            "episode_end_time": self.episode_end_time,
            "duration_ns": self.duration_ns,
            "agent_info": self.agent_info
        }

        return ret






class GameHistory:
    def __init__(self):
        self.timestamp = datetime.now()
        self.unix_ts = self.timestamp.timestamp()
        self.id = self.make_id()
        self.seed:int = None
        self.bag:list[str] = []
        self.placements:list[MinoPlacement] = []
        # Not including overflow
        self.field_dims:tuple[int,int] = (20, 10)

        # Probably not going to use this, but it's another
        # data collector so let's hold onto it.
        self.record:TetrisGameRecord = None

    def make_id(self):
        """
        Produces IDs like 240714-beeeef
        """
        hexstr = '0123456789abcdef'
        ret = ""
        ret += self.timestamp.strftime("%y%m%d") + "-"
        ret += ''.join([random.choice(hexstr) for x in range(6)])
        return ret

    def to_jsonable(self):
        ret = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "unix_ts": self.unix_ts,
            "seed": self.seed,
            "bag": self.bag,
            "placements": [x.to_jsonable() for x in self.placements],
            "field_dims": self.field_dims,
            "record": self.record.to_jsonable() if self.record is not None else {}
            }

        return ret

