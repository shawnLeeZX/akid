"""
Source for ImageNet data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import os

import tensorflow as tf
from torchvision import datasets, transforms

from akid import (
    SupervisedSource,
    ClassificationTFSource,
    StaticSource
)

from akid import backend as A
import six


"""Small library that points to a data set.

Methods of Data class:
  data_files: Returns a python list of all (sharded) data set files.
  num_examples_per_epoch: Returns the number of examples in the data set.
  num_classes: Returns the number of classes in the data set.
  reader: Return a reader for a single entry from the data set.
"""


class Dataset(six.with_metaclass(ABCMeta, object)):
    """A simple class for handling data sets."""

    def __init__(self, work_dir, name, subset):
        """Initialize dataset using a subset and the path to the data."""
        assert subset in self.available_subsets(), self.available_subsets()
        self.work_dir = work_dir
        self.name = name
        self.subset = subset

    @abstractmethod
    def num_classes(self):
        """Returns the number of classes in the data set."""
        pass
        # return 10

    @abstractmethod
    def num_examples_per_epoch(self):
        """Returns the number of examples in the data subset."""
        pass
        # if self.subset == 'train':
        #   return 10000
        # if self.subset == 'validation':
        #   return 1000

    @abstractmethod
    def download_message(self):
        """Prints a download message for the Dataset."""
        pass

    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation']

    def data_files(self):
        """Returns a python list of all (sharded) data subset files.

        Returns:
            python list of all (sharded) data set files.
        Raises:
            ValueError: if there are not data_files matching the subset.
        """
        tf_record_pattern = os.path.join(self.work_dir, '%s-*' % self.subset)
        data_files = tf.gfile.Glob(tf_record_pattern)
        if not data_files:
            print('No files found for dataset %s/%s at %s' % (self.name,
                                                              self.subset,
                                                              self.work_dir))

            self.download_message()
            exit(-1)

        return data_files

    def reader(self):
        """Return a reader for a single entry from the data set.

        See io_ops.py for details of Reader class.

        Returns:
        Reader object that reads the data set.
        """
        return tf.TFRecordReader()


class ImagenetData(Dataset):
    """ImageNet data set."""

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 1000

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data set."""
        # Bounding box data consists of 615299 bounding boxes for 544546
        # images.
        if self.subset == 'train':
            return 1281167
            if self.subset == 'validation':
                return 50000

    def download_message(self):
        """Instruction to download and extract the tarball from Flowers
        website."""

        print('Failed to find any ImageNet %s files' % self.subset)
        print('')
        print('If you have already downloaded and processed the data, then make '
              'sure to set --data_dir to point to the directory containing the '
              'location of the sharded TFRecords.\n')
        print('If you have not downloaded and prepared the ImageNet data in the '
              'TFRecord format, you will need to do this at least once. This '
              'process could take several hours depending on the speed of your '
              'computer and network connection\n')
        print('Please see README.md for instructions on how to build '
              'the ImageNet dataset using download_and_preprocess_imagenet.\n')
        print('Note that the raw data size is 300 GB and the processed data size '
              'is 150 GB. Please ensure you have at least 500GB disk space.')


class ImagenetTFSource(ClassificationTFSource):
    def __init__(self, has_super_label=True, **kwargs):
        super(ImagenetTFSource, self).__init__(**kwargs)
        self.has_super_label = has_super_label

    def _forward(self):
        """
        Read, and crop images using bounding boxes. The cropped images and its
        labels are the output of this source. In this case of validation
        samples, no cropping is done.
        """

        with tf.device('/cpu:0'):
            dataset = ImagenetData(work_dir=self.work_dir,
                                   subset="train",
                                   name="Imagenet")
            tensor_list = self._read_dataset(dataset, train=True)
            self._training_datum = tensor_list[0]
            self._training_label = tensor_list[1:]
            # Manually set shape value, which is necessary for later
            # usage. The height and width info is dynamically determined when
            # decoding jpg files, so is not available. tf.slice seems to lose
            # all shape info when one of the shape info is missing before
            # slicing.
            self._training_datum.set_shape([None, None, 3])
            for t in self._training_label:
                t.set_shape([1])

            dataset = ImagenetData(work_dir=self.work_dir,
                                   subset="validation",
                                   name="Imagenet")
            tensor_list = self._read_dataset(dataset, train=False)
            self._val_datum = tensor_list[0]
            self._val_label = tensor_list[1:]
            self._val_datum.set_shape([None, None, 3])
            for t in self._val_label:
                t.set_shape([1])

    def _read_dataset(self, dataset, train):
        """
        Given a dataset return the data tensors for images and labels.
        """
        data_files = dataset.data_files()
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        # Create filename_queue
        if train:
            filename_queue = tf.train.string_input_producer(data_files,
                                                            shuffle=True,
                                                            capacity=16)
        else:
            filename_queue = tf.train.string_input_producer(data_files,
                                                            shuffle=False,
                                                            capacity=1)
        reader = dataset.reader()
        _, example_serialized = reader.read(filename_queue)
        image_buffer, label, bbox, _ = self.parse_example_proto(
            example_serialized)
        if train:
            image = self.process_train_image(image_buffer, bbox, 0)
        else:
            image = self.decode_jpeg(image_buffer)
        if type(label) is list:
            label.insert(0, image)
            out_tensor_list = label
        else:
            out_tensor_list = [image, label]

        return out_tensor_list

    def decode_jpeg(self, image_buffer, scope=None):
        """Decode a JPEG string into one 3-D float image Tensor.

        Args:
            image_buffer: scalar string Tensor.
            scope: Optional scope for op_scope.
        Returns:
            3-D float Tensor with values ranging from [0, 1).
        """
        with tf.name_scope(values=[image_buffer], name=scope, default_name='decode_jpeg'):
            # Decode the string as an RGB JPEG.
            # Note that the resulting image contains an unknown height and
            # width that is set dynamically by decode_jpeg. In other words, the
            # height and width of image is unknown at compile-time.
            image = tf.image.decode_jpeg(image_buffer, channels=3)

            # After this point, all image pixels reside in [0,1) until the very
            # end, when they're rescaled to (-1, 1).  The various adjust_* ops
            # all require this range for dtype float.
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            return image

    def parse_example_proto(self, example_serialized):
        """Parses an Example proto containing a training example of an image.

        The output of the build_image_data.py image preprocessing script is a
        dataset containing serialized Example protocol buffers. Each Example
        proto contains the following fields:

        image/height: 462
        image/width: 581
        image/colorspace: 'RGB'
        image/channels: 3
        image/class/label: 615
        image/class/synset: 'n03623198'
        image/class/text: 'knee pad'
        image/object/bbox/xmin: 0.1
        image/object/bbox/xmax: 0.9
        image/object/bbox/ymin: 0.2
        image/object/bbox/ymax: 0.6
        image/object/bbox/label: 615
        image/format: 'JPEG'
        image/filename: 'ILSVRC2012_val_00041207.JPEG'
        image/encoded: <JPEG encoded string>

        Args:
            example_serialized: scalar Tensor tf.string containing a serialized
                Example protocol buffer.

        Returns:
            image_buffer: Tensor tf.string containing the contents of a JPEG
                file.
            label: Tensor tf.int32 containing the label.
            bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes,
                coords] where each coordinate is [0, 1) and the coordinates are
                arranged as [ymin, xmin, ymax, xmax].
            text: Tensor tf.string containing the human-readable label.
        """
        # Dense features in Example proto.
        feature_map = {
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
            'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                    default_value=-1),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                                   default_value=''),
        }
        if self.has_super_label:
            feature_map['image/class/super_label'] \
                = tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1)
        sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
        # Sparse features in Example proto.
        feature_map.update(
            {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                         'image/object/bbox/ymin',
                                         'image/object/bbox/xmax',
                                         'image/object/bbox/ymax']})

        features = tf.parse_single_example(example_serialized, feature_map)
        label = tf.cast(features['image/class/label'], dtype=tf.int32)
        if self.has_super_label:
            super_label = tf.cast(features['image/class/super_label'],
                                dtype=tf.int32)

        xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
        ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
        xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
        ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

        # Note that we impose an ordering of (y, x) just to make life
        # difficult.
        bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])

        # Force the variable number of bounding boxes into the shape
        # [1, num_boxes, coords].
        bbox = tf.expand_dims(bbox, 0)
        bbox = tf.transpose(bbox, [0, 2, 1])

        if self.has_super_label:
            return features['image/encoded'],\
                [label, super_label],\
                bbox,\
                features['image/class/text']
        else:
            return features['image/encoded'],\
                label,\
                bbox,\
                features['image/class/text']

    def process_train_image(self, image_buffer, bbox, thread_id):
        # Crop data
        image = self.decode_jpeg(image_buffer)
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # If preparing for training set, we crop the image by the
        # bounding boxes (if available).
        # if train:
        # Display the bounding box in the first thread only.
        if not thread_id:
            image_with_box = tf.image.draw_bounding_boxes(
                tf.expand_dims(image, 0), bbox)
            tf.summary.image('image_with_bounding_boxes', image_with_box)

        # A large fraction of image datasets contain a human-annotated
        # bounding box delineating the region of the image containing
        # the object of interest.  We choose to create a new bounding
        # box for the object which is a randomly distorted version of
        # the human-annotated bounding box that obeys an allowed range
        # of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the
        # bounding box is the entire image.
        sample_distorted_bounding_box \
            = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=bbox,
                min_object_covered=0.1,
                aspect_ratio_range=[0.75, 1.33],
                area_range=[0.05, 1.0],
                max_attempts=100,
                use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox \
            = sample_distorted_bounding_box
        if not thread_id:
            image_with_distorted_box = tf.image.draw_bounding_boxes(
                tf.expand_dims(image, 0), distort_bbox)
            tf.summary.image('images_with_distorted_bounding_box',
                             image_with_distorted_box)

        # Crop the image to the specified bounding box.
        image = tf.slice(image, bbox_begin, bbox_size)

        return image


class ImagenetTorchSource(StaticSource, SupervisedSource):
    def __init__(self, random_sized_crop=True, **kwargs):
        super(ImagenetTorchSource, self).__init__(**kwargs)
        self.random_sized_crop = random_sized_crop

    def _setup(self):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        A.summary.set_normalization(mean, std)

        t_list = []
        if self.random_sized_crop:
            t_list.append(transforms.RandomSizedCrop(224))
        else:
            t_list.append(transforms.Scale(256))
            t_list.append(transforms.RandomCrop(224))

        t_list.append(transforms.RandomHorizontalFlip())
        t_list.append(transforms.ToTensor())
        t_list.append(normalize)

        self.dataset = datasets.ImageFolder(
            self.work_dir + "/train",
            transforms.Compose(t_list))
        self.val_dataset = datasets.ImageFolder(
            self.work_dir + "/val",
            transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
    def _forward(self):
        pass
