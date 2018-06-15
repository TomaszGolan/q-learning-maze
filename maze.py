"""Maze - environment for simple q-learning problem"""
import random
import numpy as np
import matplotlib.pyplot as plt


class Maze:
    """
    A maze is given by a numpy 2D array of cells (0 - blocked, 1 - free)
    It is assumed that each cell is accessible no matter the starting point.
    """

    def __init__(self, maze, target=None, agent=None):
        """Initialize a maze:

        maze   -- 2D numpy array with maze
        target -- (y, x) position of exit
        agent  -- (y, x) initial agent position
        """
        self.maze = maze

        # cells available for agent
        self.free = {pos for pos, cell in np.ndenumerate(self.maze) if cell}

        # cells already visited by an agent
        self.visited = set()

        # set target's position randomly (unless valid target is provided)
        self.target = self.set_position(target)

        # set agent's position randomly (unless valid target is provided)
        self.agent = self.set_position(agent)

    def set_position(self, position):
        """Return object's position if valid or random free cell otherwise."""
        return \
            position if position in self.free \
            else random.sample(self.free, 1)[0]

    def visualize(self):
        """Plot the maze using matplotlib.pyplot.imshow."""
        # copy as floats to use with grayscale [0.,1.]
        maze_to_plot = self.maze.astype(float).copy()

        # visited in light gray
        for pos in self.visited: maze_to_plot[pos] = 0.9

        # agent in dark gray
        maze_to_plot[self.agent] = 0.75

        # target in semi-dark gray
        maze_to_plot[self.target] = 0.5

        plt.grid(True)
        plt.xticks(np.arange(0.5, self.maze.shape[0], 1), [])
        plt.yticks(np.arange(0.5, self.maze.shape[1], 1), [])
        plt.imshow(maze_to_plot, interpolation='none', cmap="gray")

        plt.show()
