# Code Skeleton - Inspired by Tensoflow RNN tutorial: ptb_word_lm.py

import numpy as np
import tensorflow as tf
import cv2
import math
import warnings

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    """ referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles/ """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
def rotationMatrixToEulerAngles(R):
    """ referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles/ """
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

# deepVO class
class deepVO(object):
    """ deepVO RCNN Model """

    def __init__(self, config, is_training=True, _only_position = True, num_frames=10):
        """ Initialization
        param: config = configuration
        param: dataset = KITTY Dataset 0th trajectory
            dataset structure:
        """
        # configuration
        self._config = config

        # Importing stacked images and ground truth
        # stacked images shape: [height, width, channels = 2]
        # pose_ground_truth shape: [num_frames, 12]
        self._input_images_stacked, pose_ground_truth = self._import_dataset(num_frames=10)

        #consinder only position
        #_only_position = True

        # Final weight layer size
        w_size = 10

        if not _only_position:
            pose_size = 6
            rotation_matrix = pose_ground_truth[:,:9]
            rotation_matrix = np.reshape(rotation_matrix[:,:9], [-1, 3, 3])
            self._pose_ground_truth = np.zeros((pose_ground_truth.shape[0], pose_size))
            self._pose_ground_truth[:,:3] = pose_ground_truth[:,9:12]
            for i in range(rotation_matrix.shape[0]):
                self._pose_ground_truth[i,3:6] = rotationMatrixToEulerAngles(rotation_matrix[i])
        else:
            pose_size = 3
            self._pose_ground_truth = pose_ground_truth[:,9:12]
            warnings.warn("Warning! Orientation data is ignored!")

        (output, _)  = self._build_rnn_graph(is_training)

        # Change the below code for different loss function based on paper
        regression_w = tf.get_variable('regression_w', shape=[w_size, pose_size], dtype=tf.float32)
        regression_b = tf.get_variable("regression_b", shape=[pose_size], dtype=tf.float32)

        pose_estimated = [tf.nn.xw_plus_b(output[i], regression_w, regression_b) for i in range(len(output))]
        pose_estimated = tf.reshape(tf.convert_to_tensor(pose_estimated), [num_frames, pose_size])
        # Use the contrib sequence loss and average over the batches
        self._loss = tf.reduce_sum(tf.square(pose_estimated - self._pose_ground_truth[:num_frames,:]))

    def _import_dataset(self, num_frames=None):
        """ Function that loads dataset
        """
        if num_frames == None:
            num_frames = 4500

        trajectory = 0
        img_stacked = []
        with open('/Users/Shrinath/visual-odometry/dataset/poses/' +  '%02d' % trajectory + '.txt') as f:
            poses_ground_truth = np.array([[float(x) for x in line.split()] for line in f])
        for i in range(num_frames):
            img1 = cv2.imread('/Users/Shrinath/visual-odometry/dataset/sequences/'+ '%02d' % trajectory + '/image_0/' +  '%06d' % i + '.png', 0)
            img2 = cv2.imread('/Users/Shrinath/visual-odometry/dataset/sequences/00/image_0/' + '%06d' % (i+1) + '.png', 0)
            width, height = img1.shape
            img1 = tf.cast(img1, dtype=tf.float32)
            img2 = tf.cast(img2, dtype=tf.float32)
            img_aug = tf.stack([img1, img2], axis=2)
            img_aug = tf.reshape(img_aug, (1, height, width, 2))
            img_stacked.append(img_aug)
        return img_stacked, poses_ground_truth

    def _get_lstm_cell(self, is_training):
        return tf.contrib.rnn.BasicLSTMCell(
                self._config.hidden_size,
                forget_bias=1.0)

    def _build_rnn_graph(self, is_training):
        """ RNN layers which connects to final output
            Build the inference graph using canonical LSTM cells.
        """
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        cell = self._get_lstm_cell(is_training)
        #cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(self._config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(self._config.batch_size, dtype = tf.float32)
        rnn_inputs = [self._cnn_layers(stacked_img) for \
                stacked_img in self._input_images_stacked]
        rnn_inputs = [tf.reshape(rnn_inputs[i],[-1, 20*6*1024]) for i in range(len(rnn_inputs))]
        outputs, state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, dtype=tf.float32)
        return outputs, state

    def _cnn_layers(self, input_layer):
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

# config class
class Config(object):
    def __init__(self, lstm_hidden_size=10, lstm_num_layers=2, batch_size=1, num_steps= 10):
        self.hidden_size = lstm_hidden_size
        self.num_layers = lstm_num_layers
        self.batch_size = batch_size
        self.num_steps = num_steps

def main():
    config = Config(lstm_hidden_size=10, lstm_num_layers=2)
    vo_train = deepVO(config)

if __name__ == "__main__":
    main()
