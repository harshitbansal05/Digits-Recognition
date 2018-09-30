import os
import tensorflow as tf

import model

tf.app.flags.DEFINE_string('checkpoint_path', './east_icdar2015_resnet_v1_50_rbox/', '')
FLAGS = tf.app.flags.FLAGS

output_node_names = 'boxes'


def detect(score_map, geo_map, score_map_thresh=0.8):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param score_map_thresh: threshhold for score map
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]

    # filter the score map
    xy_text = tf.where(score_map > score_map_thresh)
    xy_text = tf.cast(xy_text, tf.int32)
    # sort the text boxes via the y axis
    xy_sort_indices = tf.contrib.framework.argsort(xy_text[:, 0])
    xy_sort_indices = tf.expand_dims(xy_sort_indices, axis=1)
    xy_text = tf.gather_nd(xy_text, xy_sort_indices)
    
    # if xy_sort_indices.get_shape().as_list()[0] is not None:
    #     xy_text_sort = tf.zeros((0, 2))
    #     for i in range(xy_sort_indices.get_shape().as_list()[0]):
    #         a = tf.expand_dims(xy_text[xy_sort_indices[i], :], axis=0)
    #         xy_text_sort = tf.concat([xy_text_sort, a], axis=0)
    #     xy_text = xy_text_sort

    # restore the rectangle boxes
    geometry_map = tf.gather_nd(geo_map, xy_text)
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geometry_map) # N*4*2

    boxes_points = tf.reshape(text_box_restored, (-1, 8))
    boxes_angle = tf.gather_nd(score_map, xy_text)
    boxes_angle = tf.expand_dims(boxes_angle, axis=1)
    boxes = tf.concat([boxes_points, boxes_angle], axis=1)
    boxes = tf.reshape(boxes, [-1])
    pad = tf.zeros((2700))
    boxes = tf.concat((boxes, pad), axis=0)
    boxes = tf.slice(boxes, [0], [2700], name='boxes')

    return boxes


def restore_rectangle(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    
    # for angle > 0_
    angle_0_coordinates = tf.where(angle >= 0)
    angle_0_coordinates = tf.cast(angle_0_coordinates, tf.int32)
    origin_0 = tf.gather_nd(origin, angle_0_coordinates)
    d_0 = tf.gather_nd(d, angle_0_coordinates)
    angle_0 = tf.gather_nd(angle, angle_0_coordinates)

    angle_0_coordinates = tf.reshape(angle_0_coordinates, [-1])
    zero_tensor = tf.zeros_like(angle_0_coordinates)
    zero_tensor = tf.cast(zero_tensor, tf.float32)
    
    p = tf.stack([zero_tensor, -d_0[:, 0] - d_0[:, 2],
                  d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                  d_0[:, 1] + d_0[:, 3], zero_tensor,
                  zero_tensor, zero_tensor,
                  d_0[:, 3], -d_0[:, 2]], axis=0)
    p = tf.reshape(tf.transpose(p, [1, 0]), (-1, 5, 2))  # N*5*2

    rotate_matrix_x = tf.stack([tf.cos(angle_0), tf.sin(angle_0)], axis= 0)
    rotate_matrix_x = tf.transpose(rotate_matrix_x, [1, 0])
    rotate_matrix_x = tf.stack([rotate_matrix_x[:, 0], rotate_matrix_x[:, 0],
                                 rotate_matrix_x[:, 0], rotate_matrix_x[:, 0],
                                 rotate_matrix_x[:, 0], rotate_matrix_x[:, 1],
                                 rotate_matrix_x[:, 1], rotate_matrix_x[:, 1],
                                 rotate_matrix_x[:, 1], rotate_matrix_x[:, 1]], axis=0)
    rotate_matrix_x = tf.transpose(rotate_matrix_x, [1, 0])
    rotate_matrix_x = tf.reshape(rotate_matrix_x, (-1, 2, 5))
    rotate_matrix_x = tf.transpose(rotate_matrix_x, (0, 2, 1))  # N*5*2

    rotate_matrix_y = tf.stack([-tf.sin(angle_0), tf.cos(angle_0)], axis= 0)
    rotate_matrix_y = tf.transpose(rotate_matrix_y, [1, 0])
    rotate_matrix_y = tf.stack([rotate_matrix_y[:, 0], rotate_matrix_y[:, 0],
                                 rotate_matrix_y[:, 0], rotate_matrix_y[:, 0],
                                 rotate_matrix_y[:, 0], rotate_matrix_y[:, 1],
                                 rotate_matrix_y[:, 1], rotate_matrix_y[:, 1],
                                 rotate_matrix_y[:, 1], rotate_matrix_y[:, 1]], axis=0)
    rotate_matrix_y = tf.transpose(rotate_matrix_y, [1, 0])
    rotate_matrix_y = tf.reshape(rotate_matrix_y, (-1, 2, 5))
    rotate_matrix_y = tf.transpose(rotate_matrix_y, (0, 2, 1))  # N*5*2

    p_rotate_x = tf.reduce_sum(rotate_matrix_x * p, axis=2)
    p_rotate_x = tf.expand_dims(p_rotate_x, axis=2)  # N*5*1
    p_rotate_y = tf.reduce_sum(rotate_matrix_y * p, axis=2)
    p_rotate_y = tf.expand_dims(p_rotate_y, axis=2)  # N*5*1

    p_rotate = tf.concat([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

    origin_0 = tf.cast(origin_0, tf.float32)
    p3_in_origin = origin_0 - p_rotate[:, 4, :]
    new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
    new_p1 = p_rotate[:, 1, :] + p3_in_origin
    new_p2 = p_rotate[:, 2, :] + p3_in_origin
    new_p3 = p_rotate[:, 3, :] + p3_in_origin

    new_p0 = tf.expand_dims(new_p0, axis=1)
    new_p1 = tf.expand_dims(new_p1, axis=1)
    new_p2 = tf.expand_dims(new_p2, axis=1)
    new_p3 = tf.expand_dims(new_p3, axis=1)
    
    new_p_0 = tf.concat([new_p0, new_p1, new_p2, new_p3], axis=1)  # N*4*2

    # for angle < 0
    angle_1_coordinates = tf.where(angle < 0)
    angle_1_coordinates = tf.cast(angle_1_coordinates, tf.int32)
    origin_1 = tf.gather_nd(origin, angle_1_coordinates)
    d_1 = tf.gather_nd(d, angle_1_coordinates)
    angle_1 = tf.gather_nd(angle, angle_1_coordinates)

    angle_1_coordinates = tf.reshape(angle_1_coordinates, [-1])
    zero_tensor1 = tf.zeros_like(angle_1_coordinates)
    zero_tensor1 = tf.cast(zero_tensor1, tf.float32)

    p = tf.stack([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                  zero_tensor1, -d_1[:, 0] - d_1[:, 2],
                  zero_tensor1, zero_tensor1,
                  -d_1[:, 1] - d_1[:, 3], zero_tensor1,
                  -d_1[:, 1], -d_1[:, 2]], axis=0)
    p = tf.reshape(tf.transpose(p, [1, 0]), (-1, 5, 2))  # N*5*2

    rotate_matrix_x = tf.stack([tf.cos(-angle_1), -tf.sin(-angle_1)], axis= 0)
    rotate_matrix_x = tf.transpose(rotate_matrix_x, [1, 0])
    rotate_matrix_x = tf.stack([rotate_matrix_x[:, 0], rotate_matrix_x[:, 0],
                                 rotate_matrix_x[:, 0], rotate_matrix_x[:, 0],
                                 rotate_matrix_x[:, 0], rotate_matrix_x[:, 1],
                                 rotate_matrix_x[:, 1], rotate_matrix_x[:, 1],
                                 rotate_matrix_x[:, 1], rotate_matrix_x[:, 1]], axis=0)
    rotate_matrix_x = tf.transpose(rotate_matrix_x, [1, 0])
    rotate_matrix_x = tf.reshape(rotate_matrix_x, (-1, 2, 5))
    rotate_matrix_x = tf.transpose(rotate_matrix_x, (0, 2, 1))  # N*5*2

    rotate_matrix_y = tf.stack([tf.sin(-angle_1), tf.cos(-angle_1)], axis= 0)
    rotate_matrix_y = tf.transpose(rotate_matrix_y, [1, 0])
    rotate_matrix_y = tf.stack([rotate_matrix_y[:, 0], rotate_matrix_y[:, 0],
                                 rotate_matrix_y[:, 0], rotate_matrix_y[:, 0],
                                 rotate_matrix_y[:, 0], rotate_matrix_y[:, 1],
                                 rotate_matrix_y[:, 1], rotate_matrix_y[:, 1],
                                 rotate_matrix_y[:, 1], rotate_matrix_y[:, 1]], axis=0)
    rotate_matrix_y = tf.transpose(rotate_matrix_y, [1, 0])
    rotate_matrix_y = tf.reshape(rotate_matrix_y, (-1, 2, 5))
    rotate_matrix_y = tf.transpose(rotate_matrix_y, (0, 2, 1))  # N*5*2

    p_rotate_x = tf.reduce_sum(rotate_matrix_x * p, axis=2)
    p_rotate_x = tf.expand_dims(p_rotate_x, axis=2)  # N*5*1
    p_rotate_y = tf.reduce_sum(rotate_matrix_y * p, axis=2)
    p_rotate_y = tf.expand_dims(p_rotate_y, axis=2)  # N*5*1

    p_rotate = tf.concat([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

    origin_1 = tf.cast(origin_1, tf.float32)
    p3_in_origin = origin_1 - p_rotate[:, 4, :]
    new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
    new_p1 = p_rotate[:, 1, :] + p3_in_origin
    new_p2 = p_rotate[:, 2, :] + p3_in_origin
    new_p3 = p_rotate[:, 3, :] + p3_in_origin

    new_p0 = tf.expand_dims(new_p0, axis=1)
    new_p1 = tf.expand_dims(new_p1, axis=1)
    new_p2 = tf.expand_dims(new_p2, axis=1)
    new_p3 = tf.expand_dims(new_p3, axis=1)

    new_p_1 = tf.concat([new_p0, new_p1, new_p2, new_p3], axis=1)  # N*4*2

    return tf.concat([new_p_0, new_p_1], axis=0)


def freeze():
  g = tf.Graph()
  with g.as_default():
    image = tf.placeholder(tf.float32, shape=[None, None], name='input_image')
    width = tf.placeholder(tf.int32, shape=(None), name='width')
    width = tf.reshape(width, [])
    height = tf.placeholder(tf.int32, shape=(None), name='height')
    height = tf.reshape(height, [])
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    image = tf.reshape(image, [1, height, width, 3])
    image = tf.cast(image, tf.float32)

    f_score, f_geometry = model.model(image, is_training=False)
    boxes = detect(f_score, f_geometry)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    with tf.Session() as sess:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
      model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
      print('Restore from {}'.format(model_path))
      saver.restore(sess, model_path)

      # We use a built-in TF helper to export variables to constants
      print(output_node_names.split(","))
      output_graph_def = tf.graph_util.convert_variables_to_constants(
          sess, # The session is used to retrieve the weights
          g.as_graph_def(),
          output_node_names.split(",")
      ) 
      output_graph = './box_model_graph.pb'
      # Finally we serialize and dump the output graph to the filesystem
      with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


def main(_):
  freeze()


if __name__ == '__main__':
  tf.app.run(main=main)
