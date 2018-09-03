"""Builds the SVHN network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import re

import data_loader

TOWER_NAME = 'tower'

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('train_tf_records_file', 'svhn_data/train.tfrecords',
                           """Path to the train data directory.""")
tf.app.flags.DEFINE_string('extra_tf_records_file', 'svhn_data/extra.tfrecords',
                           """Path to the extra data directory.""")

# Global constants describing the SVHN data set.
IMAGE_SIZE = data_loader.IMAGE_SIZE
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = data_loader.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = data_loader.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_STEPS_PER_DECAY = 10000       # Steps after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.9  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 1e-2      # Initial learning rate.


def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def inference(images, drop_rate):
  """Build the SVHN model.
  Args:
    images: Images returned from the inputs().
    drop_rate: The rate of dropout.
  Returns:
    length_logits: Logits of length.
    digit_logits: Logits of digits.
  """
  
  # conv1
  with tf.variable_scope('conv1') as scope:
    conv = tf.layers.conv2d(images, filters=48, kernel_size=[5, 5], padding='same')
    conv1 = tf.nn.relu(conv, name=scope.name)
    _activation_summary(conv1)
  # norm1
  norm1 = tf.nn.lrn(conv1, 3, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                   name='norm1')
  # pool1
  pool1 = tf.layers.max_pooling2d(norm1, 2, 2, padding='SAME', name='pool1')
  # dropout1
  dropout1 = tf.layers.dropout(pool1, rate=drop_rate, name='dropout1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    conv = tf.layers.conv2d(dropout1, filters=64, kernel_size=[5, 5], padding='same')
    conv2 = tf.nn.relu(conv, name=scope.name)
    _activation_summary(conv2)
  # norm2
  norm2 = tf.nn.lrn(conv2, 3, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                   name='norm2')
  # pool2
  pool2 = tf.layers.max_pooling2d(norm2, 2, 1, padding='SAME', name='pool2')
  # dropout2
  dropout2 = tf.layers.dropout(pool2, rate=drop_rate, name='dropout2')

  # conv3
  with tf.variable_scope('conv3') as scope:
    conv = tf.layers.conv2d(dropout2, filters=128, kernel_size=[5, 5], padding='same')
    conv3 = tf.nn.relu(conv, name=scope.name)
    _activation_summary(conv3)
  # norm3
  norm3 = tf.nn.lrn(conv3, 3, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                   name='norm3')
  # pool3
  pool3 = tf.layers.max_pooling2d(norm3, 2, 2, padding='SAME', name='pool3')
  # dropout3
  dropout3 = tf.layers.dropout(pool3, rate=drop_rate, name='dropout3')

  # conv4
  with tf.variable_scope('conv4') as scope:
    conv = tf.layers.conv2d(dropout3, filters=160, kernel_size=[5, 5], padding='same')
    conv4 = tf.nn.relu(conv, name=scope.name)
    _activation_summary(conv4)
  # norm4
  norm4 = tf.nn.lrn(conv4, 3, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                   name='norm4')
  # pool4
  pool4 = tf.layers.max_pooling2d(norm4, 2, 1, padding='SAME', name='pool4')
  # dropout4
  dropout4 = tf.layers.dropout(pool4, rate=drop_rate, name='dropout4')

  # conv5
  with tf.variable_scope('conv5') as scope:
    conv = tf.layers.conv2d(dropout4, filters=192, kernel_size=[5, 5], padding='same')
    conv5 = tf.nn.relu(conv, name=scope.name)
    _activation_summary(conv5)
  # norm5
  norm5 = tf.nn.lrn(conv5, 3, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                   name='norm5')
  # pool5
  pool5 = tf.layers.max_pooling2d(norm5, 2, 2, padding='SAME', name='pool5')
  # dropout5
  dropout5 = tf.layers.dropout(pool5, rate=drop_rate, name='dropout5')

  # conv6
  with tf.variable_scope('conv6') as scope:
    conv = tf.layers.conv2d(dropout5, filters=192, kernel_size=[5, 5], padding='same')
    conv6 = tf.nn.relu(conv, name=scope.name)
    _activation_summary(conv6)
  # norm6
  norm6 = tf.nn.lrn(conv6, 3, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                   name='norm6')
  # pool6
  pool6 = tf.layers.max_pooling2d(norm6, 2, 1, padding='SAME', name='pool6')
  # dropout6
  dropout6 = tf.layers.dropout(pool6, rate=drop_rate, name='dropout6')

  # conv7
  with tf.variable_scope('conv7') as scope:
    conv = tf.layers.conv2d(dropout6, filters=192, kernel_size=[5, 5], padding='same')
    conv7 = tf.nn.relu(conv, name=scope.name)
    _activation_summary(conv7)
  # norm7
  norm7 = tf.nn.lrn(conv7, 3, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                   name='norm7')
  # pool7
  pool7 = tf.layers.max_pooling2d(norm7, 2, 2, padding='SAME', name='pool7')
  # dropout7
  dropout7 = tf.layers.dropout(pool7, rate=drop_rate, name='dropout7')

  # conv8
  with tf.variable_scope('conv8') as scope:
    conv = tf.layers.conv2d(dropout7, filters=192, kernel_size=[5, 5], padding='same')
    conv8 = tf.nn.relu(conv, name=scope.name)
    _activation_summary(conv8)
  # norm8
  norm8 = tf.nn.lrn(conv8, 3, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                   name='norm8')
  # pool8
  pool8 = tf.layers.max_pooling2d(norm8, 2, 1, padding='SAME', name='pool8')
  # dropout8
  dropout8 = tf.layers.dropout(pool8, rate=drop_rate, name='dropout8')

  # local9
  with tf.variable_scope('local9') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(dropout8, [-1, 4 * 4 * 192])
    weights = _variable_on_cpu('weights', [4 * 4 * 192, 3072],
                              tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
    biases = _variable_on_cpu('biases', [3072], tf.constant_initializer(0.1))
    local9 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local9)

  # local10
  with tf.variable_scope('local10') as scope:
    weights = _variable_on_cpu('weights', [3072, 3072],
                              tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
    biases = _variable_on_cpu('biases', [3072], tf.constant_initializer(0.1))
    local10 = tf.nn.relu(tf.matmul(local9, weights) + biases, name=scope.name)
    _activation_summary(local10)

  # digit length
  with tf.variable_scope('digitlength') as scope:
    weights = _variable_on_cpu('weights', [3072, 7],
                              tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
    biases = _variable_on_cpu('biases', [7], tf.constant_initializer(0.1))
    digitlength = tf.add(tf.matmul(local10, weights), biases, name='all_length')
    _activation_summary(digitlength)

  # first digit
  with tf.variable_scope('digit1') as scope:
    weights = _variable_on_cpu('weights', [3072, 11],
                              tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
    biases = _variable_on_cpu('biases', [11], tf.constant_initializer(0.1))
    digit1 = tf.matmul(local10, weights) + biases
    _activation_summary(digit1)

  # second digit
  with tf.variable_scope('digit2') as scope:
    weights = _variable_on_cpu('weights', [3072, 11],
                              tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
    biases = _variable_on_cpu('biases', [11], tf.constant_initializer(0.1))
    digit2 = tf.matmul(local10, weights) + biases
    _activation_summary(digit2)    

  # third digit
  with tf.variable_scope('digit3') as scope:
    weights = _variable_on_cpu('weights', [3072, 11],
                              tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
    biases = _variable_on_cpu('biases', [11], tf.constant_initializer(0.1))
    digit3 = tf.matmul(local10, weights) + biases
    _activation_summary(digit3)

  # fourth digit
  with tf.variable_scope('digit4') as scope:
    weights = _variable_on_cpu('weights', [3072, 11],
                              tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
    biases = _variable_on_cpu('biases', [11], tf.constant_initializer(0.1))
    digit4 = tf.matmul(local10, weights) + biases
    _activation_summary(digit4)

  # fifth digit
  with tf.variable_scope('digit5') as scope:
    weights = _variable_on_cpu('weights', [3072, 11],
                              tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
    biases = _variable_on_cpu('biases', [11], tf.constant_initializer(0.1))
    digit5 = tf.matmul(local10, weights) + biases
    _activation_summary(digit5) 

  digits_logits = tf.stack([digit1, digit2, digit3, digit4, digit5], axis=1, name='all_digits')
  return digitlength, digits_logits


def inputs(source_data):
  """Construct input for SVHN training and evaluation using the Reader ops.
  Args:
    source_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    lengths: Length Labels. 1D tensor of [batch_size] size.
    digits: Digit Labels. 2D tensor of [batch_size, 5] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.train_tf_records_file  or not FLAGS.extra_tf_records_file or not FLAGS.test_tf_records_file:
    raise ValueError('Please supply a tf_records_file')
  if source_data:
    images, lengths, digits = data_loader.inputs(filenames=[FLAGS.train_tf_records_file, FLAGS.extra_tf_records_file],
                                        batch_size=FLAGS.batch_size,
                                        num_examples_per_epoch=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
  else:
    images, lengths, digits = data_loader.inputs(filenames=[FLAGS.test_tf_records_file],
                                        batch_size=FLAGS.batch_size,
                                        num_examples_per_epoch=NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,
                                        shuffle=False)
  return images, lengths, digits


def loss_(length_logits, digits_logits, length_labels, digits_labels):
  """Add Cross entropy loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    length_logits, digits_logits: Logits from inference().
    length_labels, digits_labels: Labels from inputs(). 
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  length_labels = tf.cast(length_labels, tf.int64)
  digits_labels = tf.cast(digits_labels, tf.int64)
  length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_labels, logits=length_logits))
  digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 0], logits=digits_logits[:, 0, :]))
  digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 1], logits=digits_logits[:, 1, :]))
  digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 2], logits=digits_logits[:, 2, :]))
  digit4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 3], logits=digits_logits[:, 3, :]))
  digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 4], logits=digits_logits[:, 4, :]))

  cross_entropy_mean = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy

  tf.add_to_collection('losses', cross_entropy_mean)

  return cross_entropy_mean


def _add_loss_summaries(total_loss):
  """Add summaries for losses in SVHN model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train SVHN model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  decay_steps = NUM_STEPS_PER_DECAY

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  with tf.control_dependencies([apply_gradient_op]):
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

  return variables_averages_op
