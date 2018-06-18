"""Constants etc..."""
import numpy as np


class Settings:
    """Model's settings"""
    EXPLORE_PROB = 0.1  # the probability of exploration
    GAMMA = 0.95        # weight of next move max reward in Q function

    NOF_HIDDEN_NEURONS = 25  # in NET01 - 2 hidden layers (same size)
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 256

    NOF_EPOCHS = 100  # here epoch -> one game


class Rewards:
    """
    Rewards for actions taken by an agent:

        invalid - try to move to blocked cell or out of maze
        visited - move to already visited cell
        valid   - move to available cell
        success - exit found
    """
    INVALID, VISITED, VALID, SUCCESS = -0.75, -0.25, -0.05, 1.

    MIN_REWARDS = -25  # agent loses if total rewards < MIN_REWARDS


class Moves:
    """
    Possbile moves:

        LEFT  = 0
        RIGHT = 1
        UP    = 2
        DOWN  = 3

    Position is given by numpy 2D array indices [row, col] -> [y, x]
    """
    ALL = LEFT, RIGHT, UP, DOWN = range(4)

    @staticmethod
    def move(position, direction):
        """Return new position"""
        if direction == Moves.LEFT:
            return position[0], position[1] - 1
        elif direction == Moves.RIGHT:
            return position[0], position[1] + 1
        elif direction == Moves.UP:
            return position[0] - 1, position[1]
        elif direction == Moves.DOWN:
            return position[0] + 1, position[1]


class Mazemaps:
    """Predefined mazes"""
    MAP01 = np.array([
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 1, 0, 1],
        [0, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
    ])

    MAP02 = np.array([
        [1, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
        [1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 1, 0, 0, 1, 1]
    ])
