"""Quality functions"""
import tensorflow as tf
from settings import Moves


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
        out_size = len(Moves.ALL)
        h01_size = in_size
        h02_size = in_size

        # placeholders for features and targets
        self.x = tf.placeholder(tf.float32, [None, in_size])
        self.y = tf.placeholder(tf.float32, [None, out_size])

        # weights
        w01 = tf.Variable(tf.random_normal([in_size,  h01_size]))
        w02 = tf.Variable(tf.random_normal([h01_size, h02_size]))
        w03 = tf.Variable(tf.random_normal([h02_size, out_size]))

        # biases
        b01 = tf.Variable(tf.zeros([h01_size]))
        b02 = tf.Variable(tf.zeros([h02_size]))
        b03 = tf.Variable(tf.zeros([out_size]))

        # hidden layers
        h01 = tf.nn.relu(tf.add(tf.matmul(self.x, w01), b01))
        h02 = tf.nn.relu(tf.add(tf.matmul(h01, w02), b02))

        # output layer
        self.out = tf.add(tf.matmul(h02, w03), b03)

        self.sess.run(tf.global_variables_initializer())

    def predict(self, state):
        """Predict next move"""
        return self.sess.run(tf.argmax(self.out, 1),
                             feed_dict={self.x: state})[0]
