import tensorflow as tf
import math
import os

from datetime import datetime

import data_loader
from model import *

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = data_loader.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
BATCH_SIZE = 32


def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
  	dirname = os.path.dirname(__file__)
    dest_directory = os.path.join(dirname, 'svhn_data/train_')
    ckpt = tf.train.get_checkpoint_state(dest_directory)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / BATCH_SIZE))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * BATCH_SIZE
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)  
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(): 
  """Eval SVHN for the entire test dataset."""
  with tf.Graph().as_default() as g:
    # Get images and labels for SVHN.
    images, length_labels, digits_labels = inputs(False)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    length_logits, digits_logits = inference(images, 0.0)
    
    # Calculate predictions.
    length_labels = tf.cast(length_labels, tf.int64)
    digits_labels = tf.cast(digits_labels, tf.int64)
    length_top_k_op = tf.nn.in_top_k(length_logits, length_labels[:, 0], 1)
    digit1_top_k_op = tf.nn.in_top_k(digits_logits[:, 0, :], digits_labels[:, 0], 1)
    digit2_top_k_op = tf.nn.in_top_k(digits_logits[:, 1, :], digits_labels[:, 1], 1)
    digit3_top_k_op = tf.nn.in_top_k(digits_logits[:, 2, :], digits_labels[:, 2], 1)
    digit4_top_k_op = tf.nn.in_top_k(digits_logits[:, 3, :], digits_labels[:, 3], 1)
    digit5_top_k_op = tf.nn.in_top_k(digits_logits[:, 4, :], digits_labels[:, 4], 1)
    top_k_op = length_top_k_op & digit1_top_k_op & digit2_top_k_op & digit3_top_k_op & digit4_top_k_op & digit5_top_k_op
    
    variable_averages = tf.train.ExponentialMovingAverage(
        0.9999)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    dirname = os.path.dirname(__file__)
    dest_directory = os.path.join(dirname, 'svhn_data/eval')
    summary_writer = tf.summary.FileWriter(dest_directory, g)

    eval_once(saver, summary_writer, top_k_op, summary_op)


def main(_):
  dirname = os.path.dirname(__file__)
  dest_directory = os.path.join(dirname, 'svhn_data/eval')
  if tf.gfile.Exists(dest_directory):
    tf.gfile.DeleteRecursively(dest_directory)
  tf.gfile.MakeDirs(dest_directory)
  evaluate()


if __name__ == '__main__':
  tf.app.run(main=main)