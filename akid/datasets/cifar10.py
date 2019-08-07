"""
This module holds sources (feed source and tfrecord source) for CIFAR10
dataset. It supports using raw CIFAR10 images as input, as well as global
contrasted normalized, and ZCA whitened images (the common pre-processing done
on CIFAR dataset, starting from the Maxout paper).
"""


from __future__ import absolute_import
import os
import six.moves.cPickle as pickle

import numpy as np
import tensorflow as tf
from torchvision import datasets, transforms

from ..core.sources import (
    Source,
    InMemoryFeedSource,
    SupervisedSource,
    ClassificationTFSource,
    StaticSource
)
from .datasets import DataSet, DataSets
from six.moves import range

from akid import backend as A


class Cifar10Source(SupervisedSource):
    """
    A shared ancestor for different CIFAR10 sources. It is still an abstract
    class.
    """
    IMAGE_SIZE = 32
    SAMPLE_NUM = 50000

    def __init__(self, use_zca=False, **kwargs):
        """
        Args:
            use_zca: Boolean
                Use ZCA whitened data or not. If this is specified, the ZCA
                whitened data has to be in `work_dir` already.
        """
        super(Cifar10Source, self).__init__(**kwargs)
        self.use_zca = use_zca

    def _load_cifar10_python(self, filenames):
        """
        Load python version of Cifar10 dataset.
        """
        # Load the first batch of data to get shape info.
        filename = filenames[0]
        with open(filename, "rb") as f:
            tmp = pickle.load(f)
            data = tmp["data"]
            labels = np.array(tmp["labels"])

        # Load the rest.
        for filename in filenames[1:]:
            with open(filename, "rb") as f:
                tmp = pickle.load(f)
                data = np.append(data, tmp["data"], 1)
                labels = np.append(labels, tmp["labels"])

        data = np.reshape(data, [-1, 3, 32, 32])
        data = np.einsum("nchw->nhwc", data)

        return DataSet(data, labels)


class Cifar10FeedSource(Cifar10Source, InMemoryFeedSource):
    """
    A concrete `FeedSource` for Cifar10 dataset.
    """
    @property
    def shape(self):
        return [32, 32, 3]

    @property
    def label_shape(self):
        return [1]

    def _load(self):
        self._get_raw_data_if_not_yet()

        # Load training set into memory.
        train_filenames = [os.path.join(self.work_dir, 'cifar-10-batches-py',
                                        'data_batch_%d' % i)
                           for i in range(1, 6)]
        training_dataset = self._load_cifar10_python(train_filenames)

        test_filenames = [os.path.join(self.work_dir, 'cifar-10-batches-py',
                                       'test_batch')]
        test_dataset = self._load_cifar10_python(test_filenames)

        if self.use_zca:
            sample_num = len(train_filenames) * 10000
            imgs = np.load(os.path.join(self.work_dir,
                                        "pylearn2_gcn_whitened",
                                        "train.npy"))
            imgs = imgs.reshape([sample_num, 3, 32, 32])
            imgs = np.einsum("nchw->nhwc", imgs)
            training_dataset = DataSet(
                imgs[0:self.num_train, ...],
                training_dataset.labels[0:self.num_train])

            imgs = np.load(os.path.join(self.work_dir,
                                        "pylearn2_gcn_whitened",
                                        "test.npy"))
            imgs = imgs.reshape([10000, 3, 32, 32])
            imgs = np.einsum("nchw->nhwc", imgs)
            test_dataset = DataSet(imgs[0:self.num_val, ...],
                                   test_dataset.labels[0:self.num_val])

        return DataSets(training_dataset, test_dataset)


class Cifar10TFSource(Cifar10Source, ClassificationTFSource):
    """
    A concrete `Source` for Cifar10 dataset.
    """
    def _forward(self):
        """
        Construct input for CIFAR evaluation using the Reader ops.
        """
        if self.use_zca:
            self._read_from_tfrecord()
        else:
            self._read_from_bin()

    def _read_from_tfrecord(self):
        self._maybe_convert_to_tf()

        # Read and set up data tensors.
        filename = os.path.join(self.work_dir, 'cifar10_training.tfrecords')
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer([filename])
        self._training_datum, self._training_label \
            = self._get_sample_tensors_from_tfrecords(filename_queue)

        filename = os.path.join(self.work_dir, 'cifar10_test.tfrecords')
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer([filename])
        self._val_datum, self._val_label \
            = self._get_sample_tensors_from_tfrecords(filename_queue)

    def _maybe_convert_to_tf(self):
        """
        If tfrecords data are not available, convert it from the ZCA whitened
        data and downloaded labels.

        TODO: write the data pre-processing code so I won't need to rely on
        pylearn2.
        """
        TRAINING_TF_FILENAME = "cifar10_training"
        if not os.path.exists(
                os.path.join(self.work_dir,
                             TRAINING_TF_FILENAME + ".tfrecords")):
            # Read the numpy data in and convert it to TFRecord.
            imgs = np.load(os.path.join(self.work_dir,
                                        "pylearn2_gcn_whitened",
                                        "train.npy"))
            imgs = imgs.reshape([Cifar10Source.SAMPLE_NUM, 3, 32, 32])
            imgs = np.einsum("nchw->nhwc", imgs)

            # Load training set into memory.
            train_filenames = [os.path.join(self.work_dir,
                                            'cifar-10-batches-py',
                                            'data_batch_%d' % i)
                               for i in range(1, 6)]
            training_labels = self._load_cifar10_python(train_filenames).labels
            self._convert_to_tf(imgs, training_labels, TRAINING_TF_FILENAME)

        TEST_TF_FILENAME = "cifar10_test"
        if not os.path.exists(
                os.path.join(self.work_dir, TEST_TF_FILENAME + ".tfrecords")):
            imgs = np.load(os.path.join(self.work_dir,
                                        "pylearn2_gcn_whitened",
                                        "test.npy"))
            imgs = imgs.reshape([10000, 3, 32, 32])
            imgs = np.einsum("nchw->nhwc", imgs)
            test_filenames = [os.path.join(self.work_dir,
                                           'cifar-10-batches-py',
                                           'test_batch')]
            test_labels = self._load_cifar10_python(test_filenames).labels
            self._convert_to_tf(imgs, test_labels, TEST_TF_FILENAME)

    def _get_sample_tensors_from_tfrecords(self, filename_queue):
        """
        Read from tfrecord file and return data tensors.

        Args:
            filename_queue: tf.train.string_input_producer
                A file name queue that gives string tensor for tfrecord names.

        Returns:
            (image, label): tuple of (rank-4 tf.float32 tensor and rank-1
                            tf.int32 tensor)
                individual sample that may be later put into a batch.
        """
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image_raw': tf.FixedLenFeature(
                    [Cifar10Source.IMAGE_SIZE, Cifar10Source.IMAGE_SIZE, 3],
                    tf.float32),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['label'], tf.int32)
        image = features["image_raw"]

        return image, label

    def _read_from_bin(self):
        self._get_raw_data_if_not_yet()

        filenames = [os.path.join(self.work_dir, 'cifar-10-batches-bin',
                                  'data_batch_%d.bin' % i)
                     for i in range(1, 6)]
        datum, label = self._read_by_filenames(filenames)
        self._training_datum = datum
        self._training_label = label

        filenames = [os.path.join(self.work_dir, 'cifar-10-batches-bin',
                                  'test_batch.bin')]
        datum, label = self._read_by_filenames(filenames)
        self._val_datum = datum
        self._val_label = label

    def _read_by_filenames(self, filenames):
        for f in filenames:
            if not os.path.exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)

        # Read examples from files in the filename queue.
        read_input = self._read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        return reshaped_image, read_input.label

    def _read_cifar10(self, filename_queue):
        """Reads and parses examples from CIFAR10 data files.

        Recommendation: if you want N-way read parallelism, call this function
        N times.  This will give you N independent Readers reading different
        files & positions within those files, which will give better mixing of
        examples.

        Args:
            filename_queue: A queue of strings with the filenames to read from.

        Returns:
            An object representing a single example, with the following fields:
            height: number of rows in the result (32)
            width: number of columns in the result (32)
            depth: number of color channels in the result (3)
            key: a scalar string Tensor describing the filename & record number
                for this example.
            label: an int32 Tensor with the label in the range 0..9.
            uint8image: a [height, width, depth] uint8 Tensor with the image
                data
        """

        class CIFAR10Record(object):
            pass
        result = CIFAR10Record()

        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of
        # the input format.
        label_bytes = 1  # 2 for CIFAR-100
        result.height = 32
        result.width = 32
        result.depth = 3
        image_bytes = result.height * result.width * result.depth
        # Every record consists of a label followed by the image, with a
        # fixed number of bytes for each.
        record_bytes = label_bytes + image_bytes

        # Read a record, getting filenames from the filename_queue.  No
        # header or footer in the CIFAR-10 format, so we leave header_bytes
        # and footer_bytes at their default of 0.
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        result.key, value = reader.read(filename_queue)

        # Convert from a string to a vector of uint8 that is record_bytes long.
        record_bytes = tf.decode_raw(value, tf.uint8)

        # The first bytes represent the label, which we convert from
        # uint8->int32.
        result.label = tf.cast(
            tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

        # The remaining bytes after the label represent the image, which we
        # reshape from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(
            tf.slice(record_bytes, [label_bytes], [image_bytes]),
            [result.depth, result.height, result.width])
        # Convert from [depth, height, width] to [height, width, depth].
        result.uint8image = tf.transpose(depth_major, [1, 2, 0])

        return result


class Cifar10TorchSource(StaticSource, SupervisedSource):
    def _setup(self):
        convert = transforms.Compose([
            transforms.ToTensor(),
            # lambda x: x.permute(2, 0, 1),
            transforms.Normalize([i/255 for i in [125.3, 123.0, 113.9]],
                                 [i/255 for i in [63.0, 62.1, 66.7]]),
        ])
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Pad(4),
            transforms.RandomCrop(32),
            convert,
        ])

        self.dataset = datasets.CIFAR10(self.work_dir, train=True, download=True,
                                        transform=train_transform)
        self.val_dataset = datasets.CIFAR10(self.work_dir, train=False,
                                            transform=convert)
    def _forward(self):
        pass


class Cifar10AkidSource(Source):
    def _setup(self):
        if not hasattr(self, "_data"):
            convert = transforms.Compose([
                transforms.ToTensor(),
                # lambda x: x.permute(2, 0, 1),
                transforms.Normalize([i/255 for i in [125.3, 123.0, 113.9]],
                                    [i/255 for i in [63.0, 62.1, 66.7]]),
            ])
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Pad(4),
                transforms.RandomCrop(32),
                convert,
            ])

            self._data = datasets.CIFAR10(self.work_dir, train=True, download=True,
                                            transform=train_transform)

    def set_mode(self, mode):
        super(Cifar10AkidSource, self).set_mode(mode)

        convert = transforms.Compose([
            transforms.ToTensor(),
            # lambda x: x.permute(2, 0, 1),
            transforms.Normalize([i/255 for i in [125.3, 123.0, 113.9]],
                                 [i/255 for i in [63.0, 62.1, 66.7]]),
        ])
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Pad(4),
            transforms.RandomCrop(32),
            convert,
        ])

        if mode == A.Mode.TRAIN:
            self._data = datasets.CIFAR10(self.work_dir, train=True, download=True,
                                          transform=train_transform)
        elif mode == A.Mode.VAL:
            self._data = datasets.CIFAR10(self.work_dir, train=False,
                                          transform=convert)
        else:
            raise ValueError("Mode {} not supported yet.".format(mode))

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return len(self._data)
