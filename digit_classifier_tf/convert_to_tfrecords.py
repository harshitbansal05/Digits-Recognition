import os
import sys
import tarfile
from six.moves import urllib
import numpy as np
import h5py
import random
from PIL import Image
import tensorflow as tf


class ExampleReader(object):
  def __init__(self, path_to_image_files):
    self._path_to_image_files = path_to_image_files
    self._num_examples = len(self._path_to_image_files)
    self._example_pointer = 0

  @staticmethod
  def _get_attrs(digit_struct_mat_file, index):
    """
    Returns a dictionary which contains keys: label, left, top, width and height, each key has multiple values.
    """
    attrs = {}
    f = digit_struct_mat_file
    item = f['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
      attr = f[item][key]
      values = [f[attr.value[i].item()].value[0][0]
                for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
      attrs[key] = values
    return attrs

  @staticmethod
  def _preprocess(image, bbox_left, bbox_top, bbox_width, bbox_height):
    cropped_left, cropped_top, cropped_width, cropped_height = (int(round(bbox_left - 0.15 * bbox_width)),
                                                                int(round(bbox_top - 0.15 * bbox_height)),
                                                                int(round(bbox_width * 1.3)),
                                                                int(round(bbox_height * 1.3)))
    image = image.crop([cropped_left, cropped_top, cropped_left + cropped_width, cropped_top + cropped_height])
    image = image.resize([64, 64])
    return image

  @staticmethod
  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  @staticmethod
  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def read_and_convert(self, digit_struct_mat_file):
    """
    Read and convert to example, returns None if no data is available.
    """
    if self._example_pointer == self._num_examples:
      return None
    path_to_image_file = self._path_to_image_files[self._example_pointer]
    index = int(path_to_image_file.split('/')[-1].split('.')[0]) - 1
    self._example_pointer += 1

    attrs = ExampleReader._get_attrs(digit_struct_mat_file, index)
    label_of_digits = attrs['label']
    length = len(label_of_digits)
    if length > 5:
      # skip this example
      return self.read_and_convert(digit_struct_mat_file)

    digits = [10, 10, 10, 10, 10]   # digit 10 represents no digit
    for idx, label_of_digit in enumerate(label_of_digits):
      digits[idx] = int(label_of_digit if label_of_digit != 10 else 0)    # label 10 is essentially digit zero

    attrs_left, attrs_top, attrs_width, attrs_height = map(lambda x: [int(i) for i in x], [attrs['left'], attrs['top'], attrs['width'], attrs['height']])
    min_left, min_top, max_right, max_bottom = (min(attrs_left),
                                                min(attrs_top),
                                                max(map(lambda x, y: x + y, attrs_left, attrs_width)),
                                                max(map(lambda x, y: x + y, attrs_top, attrs_height)))
    center_x, center_y, max_side = ((min_left + max_right) / 2.0,
                                    (min_top + max_bottom) / 2.0,
                                    max(max_right - min_left, max_bottom - min_top))
    bbox_left, bbox_top, bbox_width, bbox_height = (center_x - max_side / 2.0,
                                                    center_y - max_side / 2.0,
                                                    max_side,
                                                    max_side)
    image = np.array(ExampleReader._preprocess(Image.open(path_to_image_file), bbox_left, bbox_top, bbox_width, bbox_height)).tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': ExampleReader._bytes_feature(image),
        'length': ExampleReader._int64_feature(length),
        'digits': tf.train.Feature(int64_list=tf.train.Int64List(value=digits))
    }))
    return example


def convert_to_tfrecords(path_to_dataset_dir_list, path_to_tfrecords_file_list):
  """Helper function to generate the tfrecords file from the path to 
  dataset directory. 
  Args:
    path_to_dataset_dir: path to the images directory
    path_to_tfrecords_file: path to create the tfrecords file
  Returns:
    num_examples: the number of examples in the tfrecords file  
  """  
  num_examples = 0
  writers = []

  for path_to_tfrecords_file in path_to_tfrecords_file_list:
    writer = tf.python_io.TFRecordWriter(path_to_tfrecords_file)
    writers.append(writer)

  for i, (path_to_dataset_dir, path_to_digit_struct_mat_file) in enumerate(path_to_dataset_dir_list): 
    path_to_image_files = tf.gfile.Glob(os.path.join(path_to_dataset_dir, '*.png'))
    total_files = len(path_to_image_files)
    print('%d files found in %s' % (total_files, path_to_dataset_dir))
  
    with h5py.File(path_to_digit_struct_mat_file, 'r') as digit_struct_mat_file:
      example_reader = ExampleReader(path_to_image_files)
      for index, path_to_image_file in enumerate(path_to_image_files):
        print('(%d/%d) processing %s' % (index + 1, total_files, path_to_image_file))

        example = example_reader.read_and_convert(digit_struct_mat_file)
        if example is None:
          break

        writers[i].write(example.SerializeToString())
        num_examples += 1

  for writer in writers:
    writer.close()

  return num_examples


def convert_to_tf():
  """Helper function to generate the tfrecords file for train,
  extra and test datasets.
  Args:
    nothing
  Returns:
    num_train_examples: the number of examples in the train and extra dataset
    num_extra_examples: the number of examples in the test dataset  
  """
  dirname = os.path.dirname(__file__)
  path_to_train_dir = os.path.join(dirname, 'svhn_data/train')
  path_to_extra_dir = os.path.join(dirname, 'svhn_data/extra')
  path_to_test_dir = os.path.join(dirname, 'svhn_data/test')
  path_to_train_digit_struct_mat_file = os.path.join(path_to_train_dir, 'digitStruct.mat')
  path_to_extra_digit_struct_mat_file = os.path.join(path_to_extra_dir, 'digitStruct.mat')
  path_to_test_digit_struct_mat_file = os.path.join(path_to_test_dir, 'digitStruct.mat')

  path_to_train_tfrecords_file = os.path.join(dirname, 'svhn_data/train.tfrecords')
  path_to_extra_tfrecords_file = os.path.join(dirname, 'svhn_data/extra.tfrecords')
  path_to_test_tfrecords_file = os.path.join(dirname, 'svhn_data/test.tfrecords')
  
  for path_to_file in [path_to_train_tfrecords_file, path_to_extra_tfrecords_file, path_to_test_tfrecords_file]:
    assert not os.path.exists(path_to_file), 'The file %s already exists' % path_to_,file

  print('Processing training data...')
  num_train_examples = convert_to_tfrecords([(path_to_train_dir, path_to_train_digit_struct_mat_file),
                                            (path_to_extra_dir, path_to_extra_digit_struct_mat_file)],
                                            [path_to_train_tfrecords_file, path_to_extra_tfrecords_file])
  print('Processing test data...')
  num_test_examples = convert_to_tfrecords([(path_to_test_dir, path_to_test_digit_struct_mat_file)],
                                          [path_to_test_tfrecords_file])
  print('Done')
  return num_train_examples, num_test_examples


def maybe_download_and_extract():
  """Download and extract the tarball from SVHN's website.
  Args:
    nothing
  Returns:
    nothing  
  """
  dirname = os.path.dirname(__file__)
  dest_directory = os.path.join(dirname, 'svhn_data')
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory) 
  TRAIN_DATA_URL = 'http://ufldl.stanford.edu/housenumbers/train.tar.gz'
  EXTRA_DATA_URL = 'http://ufldl.stanford.edu/housenumbers/extra.tar.gz'
  TEST_DATA_URL = 'http://ufldl.stanford.edu/housenumbers/test.tar.gz'
  DATA_URLS = [TRAIN_DATA_URL, EXTRA_DATA_URL, TEST_DATA_URL]
  
  for data_url in DATA_URLS:
    filename = data_url.split('/')[-1]
    extracted_filename = data_url.split('/')[-1].split('.')[0]
    filepath = os.path.join(dest_directory, filename)
    extracted_filepath = os.path.join(dest_directory, extracted_filename)
    print(filepath)  
    if not os.path.exists(filepath) and not os.path.exists(extracted_filepath):
      print('Here')
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
      print()
      statinfo = os.stat(filepath)
      print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, extracted_filename)
    if not os.path.exists(extracted_dir_path):
      tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main():
  maybe_download_and_extract()
  num_train_examples, num_test_examples = convert_to_tf()