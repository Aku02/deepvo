# Code Skeleton - Inspired by Tensoflow RNN tutorial: ptb_word_lm.py

import numpy as np
import tensorflow as tf
import cv2
import math
import warnings


""" Hyper Parameters for learning"""
LEARNING_RATE = 0.001
BATCH_SIZE = 5
LSTM_HIDDEN_SIZE = 1000
LSTM_NUM_LAYERS = 2
NUM_TRAIN_STEPS = 1000
TIME_STEPS = 5


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

def build_rcnn_graph(config, input_):
    """ CNN layers connected to RNN which connects to final output """

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
    rnn_inputs = tf.reshape(rnn_inputs, [-1, max_time, 20*6*1024])

    # 'outputs' is a tensor of shape [batch_size, max_time, 1000]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                       inputs=rnn_inputs,
                                       dtype=tf.float32)
    # Tensor shaped: [batch_size, max_time, cell.output_size]
    outputs = tf.unstack(outputs, max_time, axis=1)
    return outputs, state

def get_ground_6d_poses(cordinates):
    """ For 6dof pose representaion """
    pass

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

# Dataset Class
class Kitty(object):
    """ Class for manipulating Dataset"""
    def __init__(self, config, data_dir='../dataset/', isTraining=True):
        self._config = config
        self._data_dir= data_dir
        self._img_height, self._img_width = self.get_image(0,0).shape
        self._current_initial_frame = 0
        self._current_trajectory_index = 0
        self._prev_trajectory_index = 0
        self._current_train_epoch = 0
        self._current_test_epoch = 0
        self._training_trajectories = [0, 2, 8, 9]
        self._test_trajectories = [1, 3, 4, 5, 6, 7]
        if isTraining:
            self._current_trajectories = self._training_trajectories
        else:
            self._current_trajectories = self._test_trajectories
        if not config.only_position:
            self._pose_size = 6
        else:
            self._pose_size = 3

    def get_image(self, trajectory, frame_index):
        img = cv2.imread( self._data_dir + 'sequences/'+ '%02d' % trajectory + '/image_0/' +  '%06d' % frame_index + '.png', 0)
        if img is not None:
            # Subtracting mean intensity value of the corresponding image
            img = img - np.mean(img)
            if not(trajectory==0):
                img = cv2.resize(img, (self._img_width, self._img_height), fx=0, fy=0)
        return img

    def get_poses(self, trajectory):
        with open(self._data_dir + 'poses/' +  '%02d' % trajectory + '.txt') as f:
            poses = np.array([[float(x) for x in line.split()] for line in f])
        return poses

    def _set_next_trajectory(self, isTraining):
        print 'in _set_next_trajectory, current_trj_index is %d'%self._current_trajectory_index
        if (self._current_trajectory_index < len(self._current_trajectories)-1):
            self._prev_trajectory_index = self._current_trajectory_index
            self._current_trajectory_index += 1
            self._current_initial_frame = 0
        else:
            print 'New Epoch Started'
            if isTraining:
                self._current_train_epoch += 1
            else:
                self._current_test_epoch += 1
            self._prev_trajectory_index = self._current_trajectory_index
            self._current_trajectory_index = 0
            self._current_initial_frame = 0


    def get_next_batch(self, isTraining):
        """ Function that returns the batch for dataset
        """
        img_batch = []
        label_batch = []
        if isTraining:
            self._current_trajectories = self._training_trajectories
        else:
            self._current_trajectories = self._test_trajectories

        poses = self.get_poses(self._current_trajectories[self._current_trajectory_index])
        if (self.get_image(self._current_trajectories[self._current_trajectory_index], self._current_initial_frame + self._config.time_steps) is None):
            self._set_next_trajectory(isTraining)

        print('Current Trajectory is : %d'%self._current_trajectory_index)

        for j in range(self._config.batch_size):
            img_stacked_series = []
            labels_series = []
            print('In Range : %d for %d timesteps '%(self._current_initial_frame, self._config.time_steps))
            for i in range(self._current_initial_frame, self._current_initial_frame + self._config.time_steps):
                img1 = self.get_image(self._current_trajectories[self._current_trajectory_index], i)
                img2 = self.get_image(self._current_trajectories[self._current_trajectory_index], i+1)
                img_aug = np.stack([img1, img2], -1)
                img_stacked_series.append(img_aug)
                if self._pose_size == 3:
                    pose = poses[i,9:12] - poses[self._current_initial_frame,9:12]
                else:
                    pose = get_ground_6d_poses(poses[i,:])
                labels_series.append(pose)
            img_batch.append(img_stacked_series)
            label_batch.append(labels_series)
            self._current_initial_frame += self._config.time_steps
        print np.array(img_batch).shape
        img_batch = np.reshape(np.array(img_batch), [self._config.time_steps, self._config.batch_size, self._img_height, self._img_width, 2])
        label_batch = np.reshape(np.array(label_batch), [self._config.time_steps, self._config.batch_size, self._pose_size])
        return img_batch, label_batch

# Config class
class Config(object):
    """configuration of RNN """
    def __init__(self, lstm_hidden_size=1000, lstm_num_layers=2, batch_size=1, num_steps= 20, learning_rate=0.001, only_position=True, time_steps=100):
        self.hidden_size = lstm_hidden_size
        self.num_layers = lstm_num_layers
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.only_position = only_position
        self.time_steps = time_steps

def main():
    """ main function """

    # configuration
    config = Config(lstm_hidden_size=LSTM_HIDDEN_SIZE, lstm_num_layers=LSTM_NUM_LAYERS,
            time_steps=TIME_STEPS, num_steps=NUM_TRAIN_STEPS, batch_size=BATCH_SIZE)
    kitty_data = Kitty(config)
    if not config.only_position:
        pose_size = 6
    else:
        pose_size = 3
        warnings.warn("Warning! Orientation data is ignored!")

    # only for gray scale dataset, for colored channels will be 6
    height, width, channels = 376, 1241, 2

    # placeholder for input
    with tf.name_scope('input'):
        input_data = tf.placeholder(tf.float32, [config.time_steps, None, height, width, channels])
        # placeholder for labels
        labels_ = tf.placeholder(tf.float32, [config.time_steps, None, pose_size])

    with tf.name_scope('unstacked_input'):
        # Unstacking the input into list of time series
        input_ = tf.unstack(input_data, config.time_steps, 0)
        # Unstacking the labels into the time series
        pose_labels = tf.unstack(labels_, config.time_steps, 0)


    # Building the RCNN Network which
    # which returns the time series of output layers
    with tf.name_scope('RCNN'):
        (outputs, _)  = build_rcnn_graph(config, input_)

    # Output layer to compute the output
    with tf.name_scope('weights'):
        regression_w = tf.get_variable('regression_w', shape=[config.hidden_size, pose_size], dtype=tf.float32)
    with tf.name_scope('biases'):
        regression_b = tf.get_variable("regression_b", shape=[pose_size], dtype=tf.float32)

    # Pose estimate by multiplication with RCNN_output and Output layer
    with tf.name_scope('Wx_plus_b'):
        pose_estimated = [tf.nn.xw_plus_b(output_state, regression_w, regression_b) for output_state in outputs]

    # Converting the list of tensor into a tensor
    # Probably this is the part that is unnecessary and causing problems (slowing down the computations)
    # pose_estimated = tf.reshape(tf.convert_to_tensor(pose_estimated), [num_frames, pose_size])

    # Loss function for all the frames in a batch
    with tf.name_scope('loss_l2_norm'):
        losses = [pos_est_i - pos_lab_i for pos_est_i, pos_lab_i in zip(pose_estimated, pose_labels)]
        loss_op = tf.reduce_sum(tf.square(losses))
        tf.summary.scalar('loss_l2_norm', loss_op)

    #optimizer
    with tf.name_scope('train'):
        #optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate,
        #        beta1=0.9,
        #        beta2=0.999,
        #        epsilon=1e-08,
        #        use_locking=False,
        #        name='Adam')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.learning_rate)
        train_op = optimizer.minimize(loss_op)

    saver = tf.train.Saver()

    # Merge all the summeries and write them out to model_dir
    # by default ./model_dir
    merged = tf.summary.merge_all()
    #tf.reset_default_graph()
    #imported_meta = tf.train.import_meta_graph("./model_dir/model.meta")

    with tf.Session() as sess:
        #imported_meta.restore(sess, tf.train.latest_checkpoint('./model_dir/'))
        train_writer = tf.summary.FileWriter('./model_dir/train', sess.graph)
        test_writer = tf.summary.FileWriter('./model_dir/test')
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        # Run the initializer
        sess.run(init)
        #print("Optimization Finished!")
        # Training and Testing Loop
        for i in range(config.num_steps):
            print('step : %d'%i)
            if i % 10 == 0:  # Record summaries and test-set accuracy
                batch_x, batch_y = kitty_data.get_next_batch(isTraining=False)
                summary, acc = sess.run(
                        [merged, loss_op], feed_dict={input_data:batch_x, labels_:batch_y})
                test_writer.add_summary(summary, i)
                print('Accuracy at step %s: %s' % (i, acc))
            else:  # Record train set summaries, and train
                if i % 100 == 99:  # Record execution stats
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    batch_x, batch_y = kitty_data.get_next_batch(isTraining=False)
                    summary, _ = sess.run([merged, train_op],
                            feed_dict={input_data:batch_x, labels_:batch_y},
                            options=run_options,
                            run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)
                    print('Adding run metadata for', i)
                else:  # Record a summary
                    batch_x, batch_y = kitty_data.get_next_batch(isTraining=False)
                    summary, _ = sess.run(
                        [merged, train_op], feed_dict={input_data:batch_x, labels_:batch_y})
                    train_writer.add_summary(summary, i)
                    train_loss = sess.run(loss_op,
                            feed_dict={input_data:batch_x, labels_:batch_y})
                    print('Train_error at step %s: %s' % (i, train_loss))
            saver.save(sess, './model_dir/model_iter', global_step=i)
        save_path = saver.save(sess, "./model_dir/model")
        print("Model saved in file: %s" % save_path)
        print("epochs trained: " + str(kitty_data._current_train_epoch))
        train_writer.close()
        test_writer.close()


if __name__ == "__main__":
    main()

    """
    # Test Code for checking feeding mechanism
    config = Config(lstm_hidden_size=LSTM_HIDDEN_SIZE, lstm_num_layers=LSTM_NUM_LAYERS,
            time_steps=100, num_steps=NUM_TRAIN_STEPS, batch_size=BATCH_SIZE)
    kitty_data = Kitty(config)
    for i in range(100):
        batch_x, batch_y = kitty_data.get_next_batch(isTraining=False)
        height, width, channels = 376, 1241, 2
    print('epochs: %d'%kitty_data._current_train_epoch)
    with tf.name_scope('input'):
        input_data = tf.placeholder(tf.float32, [config.time_steps, None, height, width, channels])
        # placeholder for labels
        labels_ = tf.placeholder(tf.float32, [config.time_steps, None, 3])
    with tf.Session() as sess:
        sess.run([input_data, labels_], feed_dict={input_data:batch_x, labels_:batch_y})
    """
