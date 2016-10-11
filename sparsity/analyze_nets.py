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

tf.app.flags.DEFINE_integer(
    'label_offset', 0, 'VGG and Resnet models only output scores for 1000 classes')

FLAGS = tf.app.flags.FLAGS

im_height = FLAGS.eval_image_size
im_width = FLAGS.eval_image_size
im_channels = 3

top_k = 5

num_classes = 1001 - FLAGS.label_offset

checkpoint_map = { 'vgg_16': 'vgg_16.ckpt',
                   'vgg_19': 'vgg_19.ckpt',
                   'inception_v1': 'inception_v1.ckpt',
                   'inception_v2': 'inception_v2.ckpt',
                   'inception_v3': 'inception_v3.ckpt',
                   'resnet_v1_50': 'resnet_v1_50.ckpt',
                   'resnet_v1_101': 'resnet_v1_101.ckpt',
                   'resnet_v1_152': 'resnet_v1_152.ckpt',
                 }

checkpoint_file = os.path.join(FLAGS.checkpoint_path, FLAGS.model_name + '.ckpt')
if not os.path.isfile(checkpoint_file):
    print('Checkpoint file for the model does not exist')
    sys.exit(0)

# Override the default graph
with tf.Graph().as_default() as g:
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

    # Get the function for constructing the network
    network_fn = nets_factory.get_network_fn(FLAGS.model_name,
                                             num_classes,
                                             is_training=False)

    with tf.Session() as sess:
        coordinator = tf.train.Coordinator()
        # Start the image processing workers
        threads = image_producer.start(session=sess, coordinator=coordinator)
        labels, images = image_producer.get(sess)
        print labels
        # Define the model
        # Note: The network_fn itself constructs the newtork with the right
        # scope. There is no need to do it explicitly.
        logits, _ = network_fn(images)

        # Convert the logits into a probability distribution over classes
        probs = tf.nn.softmax(logits)

        # Get a function to assign variables from a checkpoint
        #print(slim.get_variables_to_restore())
        print(checkpoint_file)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file,
                                                 slim.get_model_variables())
        init_fn(sess)
        probs = sess.run([probs])
        print np.argmax(probs)
        #slim.model_analyzer.analyze_ops(g, print_info = True)
        #print(len(slim.get_model_variables()))

        # Load variabels for the model from a checkpoint

        # Stop the worker threads
        coordinator.request_stop()
        coordinator.join(threads, stop_grace_period_secs=2)
