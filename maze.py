"""Maze - environment for simple q-learning problem"""
from random import random, sample
import numpy as np
import matplotlib.pyplot as plt
from settings import Moves, Rewards


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

        # to make it easier keep exit fixed -  in bottom right corner of a maze
        self.target = (self.maze.shape[0] - 1, self.maze.shape[1] - 1)

        self.reset(agent)

    def reset(self, agent=None):
        """Reset maze - clear visited cells and reposition agent"""
        # cells already visited by an agent
        self.visited = set()

        # set agent's position randomly (unless valid target is provided)
        self.agent = self.init_position(agent)

    def init_position(self, position=None):
        """Return object's position if valid or random free cell otherwise."""
        return position if position in self.free else sample(self.free, 1)[0]

    def get_snapshot(self):
        """Get the maze in plotable format"""
        # copy as floats to use with grayscale [0.,1.]
        maze_to_plot = self.maze.astype(float).copy()

        # visited in light gray
        for pos in self.visited:
            maze_to_plot[pos] = 0.75

        # agent in dark gray
        maze_to_plot[self.agent] = 0.5

        # target in darker gray
        maze_to_plot[self.target] = 0.25

        return maze_to_plot

    def visualize(self):
        """Plot the maze using matplotlib.pyplot.imshow"""
        plt.grid(True)
        plt.xticks(np.arange(0.5, self.maze.shape[0], 1), [])
        plt.yticks(np.arange(0.5, self.maze.shape[1], 1), [])
        plt.imshow(self.get_snapshot(), interpolation='none', cmap="gray")
        plt.show()

    def get_valid_moves(self):
        """Return a list of available moves for current agent's position"""
        return {m for m in Moves.ALL if Moves.move(self.agent, m) in self.free}

    def move_agent(self, direction):
        """Make a move and return a reward"""

        # stay in the same cell if it is impossible to move in given direction
        if direction not in self.get_valid_moves():
            return Rewards.INVALID

        # keep visited cell for visualization purpose
        self.visited.add(self.agent)

        # update agent position
        self.agent = Moves.move(self.agent, direction)

        # get a score for this move
        return Rewards.VISITED if self.agent in self.visited else (
               Rewards.SUCCESS if self.agent == self.target else
               Rewards.VALID)

    def to_vector(self):
        """Return current state as a vector"""
        maze_state = self.maze.copy()
        maze_state[self.agent] = 0.5

        return maze_state.reshape(1, -1)
