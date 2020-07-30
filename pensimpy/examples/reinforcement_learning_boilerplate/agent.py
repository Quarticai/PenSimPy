import random
import numpy as np


class Agent:
    """
    This is a simple agent samples random actions.
    """

    def __init__(self, act_dim):
        self.act_dim = act_dim

    def sample_actions(self):
        return np.clip([random.uniform(-1.5, 1.5) for _ in range(self.act_dim)], -0.01, 0.01)
