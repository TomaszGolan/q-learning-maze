from random import choice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import click
from maze import Maze
from settings import *
from nets import *


@click.group()
def main():
    pass


@main.command()
@click.option('--gif_path', help="Save to file (if provided).", required=False)
def play(gif_path):
    """Play the game"""

    # initialize the game
    maze = Maze(Mazemaps.MAP1)
    game_over = False
    total_reward = 0
    nof_moves = 0

    # initialize network
    sess = tf.Session()
    model = Net01(sess, maze.maze.shape[0] * maze.maze.shape[1])

    # plot settings
    fig = plt.figure()
    plt.grid('on')
    plt.xticks(np.arange(0.5, maze.maze.shape[0], 1), [])
    plt.yticks(np.arange(0.5, maze.maze.shape[1], 1), [])

    # snapshot will be updated every move
    maze_snap = plt.imshow(maze.get_snapshot(), cmap="gray")

    def until_game_over():
        """Generates frames until game_over set to True"""
        nonlocal game_over, nof_moves

        while not game_over:
            nof_moves += 1
            yield nof_moves

    def make_a_move(*args):
        """Make a move and update maze snapshot"""
        nonlocal game_over, total_reward, model, maze

        # move and update reward
        reward = maze.move_agent(model.predict(maze.to_vector()))
        total_reward += reward

        # check end game conditions
        game_over = (reward == Rewards.SUCCESS or
                     total_reward < Rewards.MIN_REWARDS)

        plt.title("Steps: {}, total reward: {}".format(args[0], total_reward))

        # update and return snapshot
        maze_snap.set_array(maze.get_snapshot())
        return maze_snap,

    # "play the game"
    game = FuncAnimation(fig, make_a_move, frames=until_game_over(), repeat=0)

    # show or save
    plt.show() if not gif_path else game.save(gif_path, writer='imagemagick')

    sess.close()

if __name__ == "__main__":
    main()
