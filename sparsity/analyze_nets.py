import numpy as np
import os
import tensorflow as tf

import dataset
import sys
sys.path.append('../models/slim')
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', 1,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 224, 'Eval image size')

tf.app.flags.DEFINE_string(
    'val_ground_truth_labels', None, 'File which has the ground truth labels')

FLAGS = tf.app.flags.FLAGS

im_height = FLAGS.eval_image_size
im_width = FLAGS.eval_image_size
im_channels = 3

# Get the preprocessing function for the network
preprocess_fn = preprocessing_factory.get_preprocessing(
                             FLAGS.model_name,
                             is_training=False)

# Create a producer from the imagenet dataset
image_producer = dataset.ImageNetProducer(FLAGS.val_ground_truth_labels,
                                          FLAGS.dataset_dir,
                                          FLAGS.batch_size,
                                          preprocess_fn,
                                          im_height, im_width, im_channels)

# Load variabels for the model from a check point
