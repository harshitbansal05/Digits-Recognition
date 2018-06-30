"""Routine for decoding the SVHN binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

# Process images of this size. 
IMAGE_SIZE = 64

# Global constants describing the SVHN data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 202353 + 33401
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 13068


def read_svhn(filename_queue):
  """Reads and parses examples from SVHN data files.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (64)
      width: number of columns in the result (64)
      depth: number of color channels in the result (3)
      length: an int32 Tensor with the label in the range 0..5.
      digits: a [5] int32 Tensor with the digits in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class SVHNRecord(object):
    pass
  result = SVHNRecord()

  # Dimensions of the images in the SVHN dataset.
  result.height = IMAGE_SIZE
  result.width = IMAGE_SIZE
  result.depth = 3
  
  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the SVHN format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'image': tf.FixedLenFeature([], tf.string),
      'length': tf.FixedLenFeature([], tf.int64),
      'digits': tf.FixedLenFeature([5], tf.int64)
    })
  image = tf.decode_raw(features['image'], tf.uint8)
  result.uint8image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
  result.length = features['length']
  result.digits = features['digits']
  return result


def _generate_image_and_label_batch(image, length, digit, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type tf.float32.
    length: 1-D Tensor of type tf.int32.
    digit: 1-D Tensor of type tf.int32.
    min_queue_examples: int32, minimum number of samples to retain
    in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    lengths: Length Labels. 1D tensor of [batch_size] size.
    digits: Digit Labels. 2D tensor of [batch_size, 5] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, lengths, digits = tf.train.shuffle_batch(
        [image, length, digit],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, lengths, digits = tf.train.batch(
        [image, length, digit],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, lengths, digits


def inputs(filenames, batch_size, num_examples_per_epoch, shuffle=True):
  """Construct input for SVHN training using the Reader ops.
  Args:
    filenames: List of the SVHN TF record files.
    batch_size: Number of images per batch.
    num_examples_per_epoch: The number of examples per epoch.
    shuffle: It indicates whether to shuffle the queue.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    lengths: Length Labels. 1D tensor of [batch_size] size.
    digits: Digit Labels. 2D tensor of [batch_size, 5] size.
  """

  dirname = os.path.dirname(__file__)
  for i, filename in enumerate(filenames):
    filenames[i] = os.path.join(dirname, filename)

  for filename in filenames:  
    if not tf.gfile.Exists(filename):
      raise ValueError('Failed to find file: ' + file_name)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)
  
  with tf.name_scope('data_augmentation'):
    # Read examples from files in the filename queue.
    read_input = read_svhn(filename_queue)
    image = tf.image.convert_image_dtype(read_input.uint8image, dtype=tf.float32)
    image = tf.multiply(tf.subtract(image, 0.5), 2)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.random_crop(image, [54, 54, 3])
    
    length = tf.reshape(read_input.length, [1])
    digits = tf.reshape(read_input.digits, [5])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d SVHM images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(image, length,
                                         digits, min_queue_examples,
                                         batch_size, shuffle)