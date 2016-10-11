'''Utility functions and classes for handling image datasets.'''

import os.path as osp
import numpy as np
import tensorflow as tf

class ImageProducer(object):
    '''
    Loads and processes batches of images in parallel.
    '''

    def __init__(self, image_paths, height, width, channels,
                 batch_size, preprocess_fn, num_concurrent=4, labels=None):
        # The data specifications describe how to process the image
        self.batch_size = batch_size
        # A list of full image paths
        self.image_paths = image_paths
        # An optional list of labels corresponding to each image path
        self.labels = labels
        # A boolean flag per image indicating whether its a JPEG or PNG
        self.extension_mask = self.create_extension_mask(self.image_paths)
        # Preprocessing function to apply to each image
        self.preprocess_fn = preprocess_fn
        # Output sizes of the sizes produced
        self.height = height
        self.width = width
        self.channels = channels
        # Create the loading and processing operations
        self.setup(batch_size=batch_size, num_concurrent=num_concurrent)

    def setup(self, batch_size, num_concurrent):
        # Validate the batch size
        num_images = len(self.image_paths)
        batch_size = min(num_images, batch_size or self.batch_size)
        if num_images % batch_size != 0:
            raise ValueError(
                'The total number of images ({}) must be divisible by the batch size ({}).'.format(
                    num_images, batch_size))
        self.num_batches = num_images / batch_size

        # Create a queue that will contain image paths (and their indices and extension indicator)
        self.path_queue = tf.FIFOQueue(capacity=num_images,
                                       dtypes=[tf.int32, tf.bool, tf.string],
                                       name='path_queue')

        # Enqueue all image paths, along with their indices
        indices = tf.range(num_images)
        self.enqueue_paths_op = self.path_queue.enqueue_many([indices, self.extension_mask,
                                                              self.image_paths])
        # Close the path queue (no more additions)
        self.close_path_queue_op = self.path_queue.close()

        # Create an operation that dequeues a single path and returns the image and its index
        (idx, processed_image) = self.process()

        # Create a queue that will contain the processed images (and their indices)
        processed_queue = tf.FIFOQueue(capacity=int(np.ceil(num_images / float(num_concurrent))),
                                       dtypes=[tf.int32, tf.float32],
                                       shapes=[(), processed_image.get_shape()],
                                       name='processed_queue')

        # Enqueue the processed image and path
        enqueue_processed_op = processed_queue.enqueue([idx, processed_image])

        # Create a dequeue op that fetches a batch of processed images off the queue
        self.dequeue_op = processed_queue.dequeue_many(batch_size)

        # Create a queue runner to perform the processing operations in parallel
        num_concurrent = min(num_concurrent, num_images)
        self.queue_runner = tf.train.QueueRunner(processed_queue,
                                                 [enqueue_processed_op] * num_concurrent)

    def start(self, session, coordinator, num_concurrent=4):
        '''Start the processing worker threads.'''
        # Queue all paths
        session.run(self.enqueue_paths_op)
        # Close the path queue
        session.run(self.close_path_queue_op)
        # Start the queue runner and return the created threads
        return self.queue_runner.create_threads(session, coord=coordinator, start=True)

    def get(self, session):
        '''
        Get a single batch of images along with their indices. If a set of labels were provided,
        the corresponding labels are returned instead of the indices.
        '''
        (indices, images) = session.run(self.dequeue_op)
        if self.labels is not None:
            labels = [self.labels[idx] for idx in indices]
            return (labels, images)
        return (indices, images)

    def batches(self, session):
        '''Yield a batch until no more images are left.'''
        for _ in xrange(self.num_batches):
            yield self.get(session=session)

    def load_image(self, image_path, is_jpeg):
        # Read the file
        file_data = tf.read_file(image_path)
        # Decode the image data
        img = tf.cond(
            is_jpeg,
            lambda: tf.image.decode_jpeg(file_data, channels = self.channels),
            lambda: tf.image.decode_png(file_data, channels = self.channels))

        # TODO: check if this is necessary
        #if self.data_spec.expects_bgr:
            # Convert from RGB channel ordering to BGR
            # This matches, for instance, how OpenCV orders the channels.
        #    img = tf.reverse(img, [False, False, True])
        return img

    def process(self):
        # Dequeue a single image path
        idx, is_jpeg, image_path = self.path_queue.dequeue()
        # Load the image
        img = self.load_image(image_path, is_jpeg)
        # Process the image
        processed_img = self.preprocess_fn(img, self.height, self.width)
        # Return the processed image, along with its index
        return (idx, processed_img)

    @staticmethod
    def create_extension_mask(paths):

        def is_jpeg(path):
            extension = osp.splitext(path)[-1].lower()
            if extension in ('.jpg', '.jpeg'):
                return True
            if extension != '.png':
                raise ValueError('Unsupported image format: {}'.format(extension))
            return False

        return [is_jpeg(p) for p in paths]

    def __len__(self):
        return len(self.image_paths)


class ImageNetProducer(ImageProducer):

    def __init__(self, val_path, data_path, batch_size, preprocess_fn,
                 height, width, channels):
        # Read in the ground truth labels for the validation set
        # The get_ilsvrc_aux.sh in Caffe's data/ilsvrc12 folder can fetch a copy of val.txt
        gt_lines = open(val_path).readlines()
        gt_pairs = [line.split() for line in gt_lines]
        # Get the full image paths
        # You will need a copy of the ImageNet validation set for this.
        image_paths = [osp.join(data_path, p[0]) for p in gt_pairs]
        # The corresponding ground truth labels
        labels = np.array([int(p[1]) for p in gt_pairs])
        # Initialize base
        super(ImageNetProducer, self).__init__(image_paths,
                                               height, width, channels,
                                               batch_size, preprocess_fn,
                                               labels = labels)
