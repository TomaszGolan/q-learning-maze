from random import choice, sample
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import click
from maze import Maze
from settings import *
from nets import *
from history import History


def print_summary(i, recent):
    """Print summary after i-th game"""
    print("Game {} ({}). In last {} games: {}% wins".format(
        i,
        "WON" if recent[-1] else "LOST",
        len(recent),
        100. * recent.count(True) / len(recent))
        )


@click.group()
def main():
    pass


@main.command()
@click.option('--save', help="Tensorflow snapshot name", required=True)
def train(save):
    """Train a model and save weights."""
    maze = Maze(Mazemaps.MAP01)
    history = History()
    sess = tf.Session()
    model = Net01(sess, maze.maze.size)
    last_results = deque(maxlen=5)  # last 5 results (False - lost, True - win)

    # play NOF_EPOCHS times
    for _ in range(Settings.NOF_EPOCHS):
        state = maze.to_vector()  # get initial maze state
        game_over = False
        total_reward = 0

        # play until exit is found or total reward < min_reward
        while not game_over:
            prev_state = state  # save current state before move

            # move according to model prediction
            # or take random move with given probability of exploration
            if np.random.rand() < Settings.EXPLORE_PROB:
                action = sample(maze.get_valid_moves(), 1)[0]
            else:
                action = model.predict(prev_state)

            # move an agent and update reward
            reward = maze.move_agent(action)
            total_reward += reward

            # check game over conditions
            game_over = (reward == Rewards.SUCCESS or
                         total_reward < Rewards.MIN_REWARDS)

            # save new maze state
            state = maze.to_vector()

            # save this episode into history
            history.save([prev_state, action, reward, state, game_over])

            # get next batch for training
            inputs, targets = get_training_data(model, history)

            # traing a model
            model.training(inputs, targets)

        last_results.append(reward == Rewards.SUCCESS)
        print_summary(_, last_results)
        maze.reset()

    # save the model and close tensorflow
    tf.train.Saver().save(sess, "./" + save)
    sess.close()


@main.command()
@click.option('--save', help="Tensorflow snapshot name", required=True)
@click.option('--gif_path', help="Save to file (if provided).", required=False)
def play(save, gif_path):
    """Play the game"""

    # initialize the game
    maze = Maze(Mazemaps.MAP01, agent=(0, 1))
    game_over = False
    total_reward = 0
    nof_moves = -1

    # initialize network
    sess = tf.Session()
    model = Net01(sess, maze.maze.size)
    tf.train.Saver().restore(sess, save)

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

        # to get first frame in anim
        if nof_moves == 0:
            return maze_snap,

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
    game = FuncAnimation(fig, make_a_move, frames=until_game_over(), repeat=0,
                         interval=1000)

    # show or save
    plt.show() if not gif_path else game.save(gif_path, writer='imagemagick')

    sess.close()

if __name__ == "__main__":
    main()
