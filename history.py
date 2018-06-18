"""Keep track on last moves made by network (used for training)"""
from collections import deque
from random import sample


class History:
    """Each entry contains:

    state      -- maze state
    action     -- agent's move
    reward     -- reward got for action
    next_state -- maze state after action
    game_over  -- True if target or min_reward reached
    """

    def __init__(self, N=10000):
        """Initialize history placeholder for N entries"""
        self.data = deque(maxlen=N)

    def save(self, entry):
        """entry = [state, action, reward, next_state, game_over]"""
        self.data.append(entry)

    def get_data(self, N):
        """Get random N entries from history"""
        return sample(self.data, min(len(self.data), N))
