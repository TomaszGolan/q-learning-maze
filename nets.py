"""Quality functions"""
import tensorflow as tf
import numpy as np
from settings import Moves, Settings


class Net01:
    """My first attempt to approximate Q with NN"""

    def __init__(self, session, in_size, snapshot=None):
        """Create a graph for NN

        session    -- tensorflow session
        input_size -- input vector size (maze width x maze height)
        snapshot   -- path to saved model
        """
        self.sess = session

        # layers size
        self.in_size = in_size
        self.out_size = len(Moves.ALL)
        h01_size = Settings.NOF_HIDDEN_NEURONS
        h02_size = Settings.NOF_HIDDEN_NEURONS

        # placeholders for features and targets
        self.x = tf.placeholder(tf.float32, [None, in_size])
        self.y = tf.placeholder(tf.float32, [None, self.out_size])

        # weights
        w01 = tf.Variable(tf.random_normal([in_size,  h01_size]))
        w02 = tf.Variable(tf.random_normal([h01_size, h02_size]))
        w03 = tf.Variable(tf.random_normal([h02_size, self.out_size]))

        # biases
        b01 = tf.Variable(tf.zeros([h01_size]))
        b02 = tf.Variable(tf.zeros([h02_size]))
        b03 = tf.Variable(tf.zeros([self.out_size]))

        # hidden layers
        h01 = tf.nn.relu(tf.add(tf.matmul(self.x, w01), b01))
        h02 = tf.nn.relu(tf.add(tf.matmul(h01, w02), b02))

        # output layer
        self.out = tf.add(tf.matmul(h02, w03), b03)

        # training
        loss = tf.reduce_mean(tf.losses.mean_squared_error(
            labels=self.y, predictions=self.out))

        self.train = \
            tf.train.AdamOptimizer(Settings.LEARNING_RATE).minimize(loss)

        self.sess.run(tf.global_variables_initializer())

    def predict(self, state):
        """Predict next move"""
        return self.sess.run(tf.argmax(self.out, 1),
                             feed_dict={self.x: state})[0]

    def maxQ(self, state):
        """Get max possible quality function value (for the next move)"""
        return np.max(self.sess.run(self.out, feed_dict={self.x: state})[0])

    def inference(self, state):
        """Get network output"""
        return self.sess.run(self.out, feed_dict={self.x: state})[0]

    def training(self, inputs, targets):
        self.sess.run(self.train, feed_dict={self.x: inputs, self.y: targets})


def get_training_data(network, history):
    """Prepare next batch of training data"""
    inputs = np.zeros((Settings.BATCH_SIZE, network.in_size))
    targets = np.zeros((Settings.BATCH_SIZE, network.out_size))

    # loop over random episodes from history
    for i, entry in enumerate(history.get_data(Settings.BATCH_SIZE)):
        state, action, reward, next_state, game_over = entry
        inputs[i] = state
        targets[i] = network.inference(state)
        if game_over:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + \
                                 Settings.GAMMA * network.maxQ(next_state)

    return inputs, targets
