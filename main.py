# Code Skeleton - Inspired by Tensoflow RNN tutorial: ptb_word_lm.py

import numpy as np
import tensorflow as tf
import cv2
import math
import warnings


""" Hyper Parameters for learning"""
LEARNING_RATE = 0.001
BATCH_SIZE = 1
LSTM_HIDDEN_SIZE = 1000
LSTM_NUM_LAYERS = 2
NUM_TRAIN_STEPS = 1000


def isRotationMatrix(R):
    """ Checks if a matrix is a valid rotation matrix
        referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    """ calculates rotation matrix to euler angles
        referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles
    """
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def get_lstm_cell(config, is_training):
    return tf.contrib.rnn.BasicLSTMCell(
            config.hidden_size,
            forget_bias=1.0)

def build_rcnn_graph(config, input_ ,is_training):
    """ CNN layers connected to RNN which connects to final output """

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    cell = get_lstm_cell(config, is_training)
    # create 2 LSTMCells
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [config.hidden_size, config.hidden_size]]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    rnn_inputs = [cnn_layers(stacked_img) for \
            stacked_img in input_]

    # Flattening the final convolution layers to feed them into RNN
    rnn_inputs = [tf.reshape(rnn_inputs[i],[-1, 20*6*1024]) for i in range(len(rnn_inputs))]

    max_time = len(rnn_inputs)
    rnn_inputs = tf.convert_to_tensor(rnn_inputs)
    rnn_inputs = tf.reshape(rnn_inputs, [config.batch_size, max_time, 20*6*1024])
    # 'outputs' is a tensor of shape [batch_size, max_time, 1000]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                       inputs=rnn_inputs,
                                       dtype=tf.float32)
    return outputs, state

def cnn_layers(input_layer):
        """ input: input_layer of concatonated images (img, img_next) where \
                shape of each imgae is (1280, 384, 3)
            output: 6th convolutional layer
            The structure of the CNN is inspired by the network for optical flow estimation
                in A. Dosovitskiy, P. Fischery, E. Ilg, C. Hazirbas, V. Golkov, P. van der
                Smagt, D. Cremers, T. Brox et al Flownet: Learning optical flow
                with convolutional networks, in Proceedings of IEEE International
                Conference on Computer Vision (ICCV)
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
        conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=128,
                kernel_size=[5, 5],
                padding ="same",
                strides=2,
                activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(
                inputs=conv2,
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
        conv4 = tf.layers.conv2d(
                inputs=conv3_1,
                filters=512,
                kernel_size=[3, 3],
                padding ="same",
                strides=2,
                activation=tf.nn.relu)
        conv4_1 = tf.layers.conv2d(
                inputs=conv4,
                filters=512,
                kernel_size=[3, 3],
                padding ="same",
                strides=1,
                activation=tf.nn.relu)
        conv5 = tf.layers.conv2d(
                inputs=conv4_1,
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
        output = tf.layers.conv2d(
                inputs=conv5_1,
                filters=1024,
                kernel_size=[3, 3],
                padding ="same",
                strides=2)
        """ The output is connected to RNN
        """
        return output

def import_dataset(num_frames=None, pose_size=3,trajectory=0):
    """ Function that loads dataset
    """
    if num_frames == None:
        num_frames = 4500
    img1 = cv2.imread('/Users/Shrinath/visual-odometry/dataset/sequences/'+ '%02d' % trajectory + '/image_0/' +  '%06d' % 0 + '.png', 0)
    height, width = img1.shape
    img_stacked = np.zeros((num_frames, 1, height, width, 2))
    with open('/Users/Shrinath/visual-odometry/dataset/poses/' +  '%02d' % trajectory + '.txt') as f:
        poses_ground_truth = np.array([[float(x) for x in line.split()] for line in f])
        labels = poses_ground_truth[:,:pose_size]
    for i in range(num_frames):
        img1 = cv2.imread('/Users/Shrinath/visual-odometry/dataset/sequences/'+ '%02d' % trajectory + '/image_0/' +  '%06d' % i + '.png', 0)
        img2 = cv2.imread('/Users/Shrinath/visual-odometry/dataset/sequences/'+ '%02d' % trajectory + '/image_0/' +  '%06d' % (i+1) + '.png', 0)
        img1 = np.reshape(img1, [height, width, 1])
        img2 = np.reshape(img2, [height, width, 1])
        img_aug = np.concatenate([img1, img2], axis=2)
        img_stacked[i, 0, :, :, :] = img_aug
    return img_stacked, labels

# Config class
class Config(object):
    """configuration of RNN """
    def __init__(self, lstm_hidden_size=1000, lstm_num_layers=2, batch_size=1, num_steps= 20, learning_rate=0.001, only_position=True):
        self.hidden_size = lstm_hidden_size
        self.num_layers = lstm_num_layers
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.only_position = only_position

def main(num_frames=10, is_training=True):
    config = Config(lstm_hidden_size=LSTM_HIDDEN_SIZE, lstm_num_layers=LSTM_NUM_LAYERS)
    # configuration
    timesteps = num_frames
    if not config.only_position:
        pose_size = 6
    else:
        pose_size = 3
        warnings.warn("Warning! Orientation data is ignored!")

    # only for gray scale dataset, for colored channels will be 6
    height, width, channels = 376, 1241, 2

    # placeholder for input
    input_data = tf.placeholder(tf.float32, [timesteps, None, height, width, channels])
    input_ = tf.unstack(input_data, timesteps, 0)
    # placeholder for labels
    labels_ = tf.placeholder(tf.float32, [None, pose_size])

    # Building the RCNN Network
    (output, _)  = build_rcnn_graph(config, input_, is_training)

    # Output layer to compute the output
    regression_w = tf.get_variable('regression_w', shape=[config.hidden_size, pose_size], dtype=tf.float32)
    regression_b = tf.get_variable("regression_b", shape=[pose_size], dtype=tf.float32)

    # Pose estimate by multiplication with RCNN_output and Output layer
    pose_estimated = [tf.nn.xw_plus_b(output[i], regression_w, regression_b) for i in range(output.shape[0])]
    pose_estimated = tf.reshape(tf.convert_to_tensor(pose_estimated), [num_frames, pose_size])

    print pose_estimated.shape
    # Loss function for all the frames in a batch
    loss_op = tf.reduce_sum(tf.square(pose_estimated - labels_[:num_frames,:]))

    #optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Start training
    with tf.Session() as sess:
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        # Run the initializer
        sess.run(init)
        for step in range(1, config.num_steps+1):
            batch_x, batch_y = import_dataset(num_frames=num_frames, pose_size=pose_size)
            print batch_x.shape
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={input_data: batch_x, labels_: batch_y})
            if step % 200 == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss = sess.run(loss_op, feed_dict={input_data: batch_x,
                    labels_: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= ")

        print("Optimization Finished!")

if __name__ == "__main__":
    main()
