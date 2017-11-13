# Code Skeleton - Inspired by Tensoflow RNN tutorial: ptb_word_lm.py

import numpy as np
import tensorflow as tf

class deepVOInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)

class deepVO(object):
    """ deepVO RCNN Model """

    def __init__(self, is_training, config, input_):
        """ Initialization
        """
        pass

    def _cnn_layers(self, input_layer):
        """ input: input_layer of concatonated images (img, img_next) where \
                shape of each imgae is (1280, 384, 3)
            output: 6th convolutional layer

            The structure of the CNN is inspired by the network for optical flow estimation
                in A. Dosovitskiy, P. Fischery, E. Ilg, C. Hazirbas, V. Golkov, P. van der
                Smagt, D. Cremers, T. Brox et al., “Flownet: Learning optical flow
                with convolutional networks,” in Proceedings of IEEE International
                Conference on Computer Vision (ICCV). IEEE, 2015, pp. 2758–2766.
        """
        # input_layer size [1280, 384, 6]
        # Convolutional Layer #1
        # Computes 32 features using a 7x7 filter with ReLU activation.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 1280, 384, 1]
        # Output Tensor Shape: [batch_size, 1280, 384, 32]
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=64,
            kernel_size=[7, 7],
            padding="same",
            strides=2,
            activation=tf.nn.relu)
        # Pooling Layer #1
        # First max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 1280, 384, 64]
        # Output Tensor Shape: [batch_size, 640, 192, 64]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=128,
                kernel_size=[5, 5],
                padding ="same",
                strides=2,
                activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=256,
                kernel_size=[5, 5],
                padding ="same",
                strides=2,
                activation=tf.nn.relu)
        conv3_1 = tf.layers.conv2d(
                inputs=conv3,
                filters=256,
                kernel_size=[3, 3],
                padding ="same",
                strides=1,
                activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3_1, pool_size=[2, 2], strides=2)
        conv4 = tf.layers.conv2d(
                inputs=pool3,
                filters=512,
                kernel_size=[3, 3],
                padding ="same",
                strides=2,
                activation=tf.nn.relu)
        conv4_1 = tf.layers.conv2d(
                inputs=conv3,
                filters=512,
                kernel_size=[3, 3],
                padding ="same",
                strides=1,
                activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(inputs=conv4_1, pool_size=[2, 2], strides=2)
        conv5 = tf.layers.conv2d(
                inputs=pool4,
                filters=512,
                kernel_size=[3, 3],
                padding ="same",
                strides=2,
                activation=tf.nn.relu)
        conv5_1 = tf.layers.conv2d(
                inputs=conv5,
                filters=512,
                kernel_size=[3, 3],
                padding ="same",
                strides=1,
                activation=tf.nn.relu)
        pool5 = tf.layers.max_pooling2d(inputs=conv5_1, pool_size=[2, 2], strides=2)
        output = tf.layers.conv2d(
                inputs=pool5,
                filters=1024,
                kernel_size=[3, 3],
                padding ="same",
                strides=2)
        """ The output is connected to RNN
        """
        return output


    def _preprocess_data(self):
        """ Preprosses:
                1. reshaping raw images (as per in the paper)
                2. creating time series list of reshaped images
                3. creating time series list of ground truth poses
        """
        pass

    def _get_lstm_cell(self, config, is_training):
        if config.rnn_mode == BASIC:
            return tf.contrib.rnn.BasicLSTMCell(
                config.hidden_size,
                forget_bias=0.0,
                state_is_tuple=True,
                reuse=not is_training)
        if config.rnn_mode == BLOCK:
            return tf.contrib.rnn.LSTMBlockCell(
                config.hidden_size, forget_bias=0.0)
        raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

    def _rnn_layers(self):
        """ RNN layers which connects to final output
            Build the inference graph using canonical LSTM cells.
        """
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        cell = self._get_lstm_cell(config, is_training)
        if is_training and config.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=config.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell(
            [cell for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(config.batch_size, data_type())
        state = self._initial_state
        inputs = tf.unstack(inputs, num=num_steps, axis=1)
        outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
                                    initial_state=self._initial_state)

def import_dataset(self, data_dir=None):
    """ Function that loads dataset
    """
    pass

def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to data directory")

    raw_data = _import_dataset(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data


if __name__ == "__main__":
    tf.app.run()
