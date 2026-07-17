# Library imports
import collections

# Local imports
from config import AGENT_SPECS

class ReplayBuffer():
    """Replay buffer system allowing random sampling of a large training-data pool."""
    def __init__(self):
        self.capacity = AGENT_SPECS["replay_buffer_capacity"]
        self.buffer = collections.deque(maxlen=self.capacity)
        