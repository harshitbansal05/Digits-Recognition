import os
import tensorflow as tf

from model import *

output_node_names = 'final_logits'
IMAGE_SIZE = 64


def freeze():
  g = tf.Graph()
  with g.as_default():
      
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE * 3], name="image_placeholder")  
    image = tf.reshape(image, (IMAGE_SIZE, IMAGE_SIZE, 3))
    image = tf.cast(image, tf.float32)

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.multiply(tf.subtract(image, 0.5), 2)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.random_crop(image, [54, 54, 3])
    images = tf.reshape(image, (1, 54, 54, 3))
    
    length_logits, digits_logits = inference(images, 0.0)

    length_logits = tf.reshape(length_logits, [-1])
    length = tf.argmax(length_logits, axis=0, name='final_length')
    length = tf.cast(length, tf.int32)  
    length = tf.reshape(length, [-1])    

    digits_logits = tf.reshape(digits_logits, (5, 11))
    digits = tf.argmax(digits_logits, axis=1, name='final_digits')
    digits = tf.cast(digits, tf.int32)
    digits = tf.reshape(digits, [-1])

    logits = tf.concat([length, digits], axis=0, name=output_node_names)
    variable_averages = tf.train.ExponentialMovingAverage(0.9999)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)  

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
        print('Done')  
      else:
        print('No checkpoint file found')
      
      # We use a built-in TF helper to export variables to constants
      output_graph_def = tf.graph_util.convert_variables_to_constants(
          sess, # The session is used to retrieve the weights
          g.as_graph_def(),
          output_node_names.split(",")
      ) 
      dirname = os.path.dirname(__file__)
      dest_directory = os.path.join(dirname, 'svhn_data/model')
      output_graph = os.path.join(dest_directory, 'svhn_model_graph.pb')
      # Finally we serialize and dump the output graph to the filesystem
      with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))  


def main(_):
  dirname = os.path.dirname(__file__)
  dest_directory = os.path.join(dirname, 'svhn_data/model')
  if tf.gfile.Exists(dest_directory):
    tf.gfile.DeleteRecursively(dest_directory)
    tf.gfile.MakeDirs(dest_directory)
  freeze()


if __name__ == '__main__':
  tf.app.run(main=main)