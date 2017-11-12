# First task is to build a CNN, which will be fed to RNN.


import numpy as np
import tensorflow as tf

def cnn_layers(features):
    """ input: concatonated images (img, img_next) where \
            shape of each imgae is (1280, 384, 3)
    """
    # input_layer size [1280, 384, 6]
    input_layer = features["stacked_images"]
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

    conv6 = tf.layers.conv2d(
            inputs=pool5,
            filters=1024,
            kernel_size=[3, 3],
            padding ="same",
            strides=2,
            activation=tf.nn.relu)

    return conv6


