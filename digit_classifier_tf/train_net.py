from datetime import datetime
import time

import os
import tensorflow as tf

import convert_to_tfrecords
from model import *

BATCH_SIZE = 32
NUM_STEPS = 50000

def train_():
  """Train SVHN for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for SVHN.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, length_labels, digits_labels = model.inputs(True)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    length_logits, digits_logits = inference(images, 0.2)

    # Calculate loss.
    loss = loss_(length_logits, digits_logits, length_labels, digits_labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % 10 == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = 10 * BATCH_SIZE / duration
          sec_per_batch = float(duration / 10)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    dirname = os.path.dirname(__file__)
    dest_directory = os.path.join(dirname, 'svhn_data/train_')
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=dest_directory,
        hooks=[tf.train.StopAtStepHook(last_step=NUM_STEPS),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=False)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(_):
  convert_to_tfrecords.main()
  dirname = os.path.dirname(__file__)
  dest_directory = os.path.join(dirname, 'svhn_data/train_')
  if tf.gfile.Exists(dest_directory):
    tf.gfile.DeleteRecursively(dest_directory)
    tf.gfile.MakeDirs(dest_directory)
  train_()


if __name__ == '__main__':
  tf.app.run(main=main)