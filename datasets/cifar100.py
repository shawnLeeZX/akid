"""
This module holds sources for CIFAR100 dataset. Only `TFSource` type source are
implemented for now.
"""
import os

import cPickle as pickle
import numpy as np
import tensorflow as tf

from ..core.sources import ClassificationTFSource
from .datasets import DataSet


class Cifar100TFSource(ClassificationTFSource):
    """
    A concrete `Source` for Cifar100 dataset. This class provides global
    contrast normalized, then ZCA whitened images using tfrecords..
    """
    SAMPLE_NUM = 50000

    def _load_cifar100_python(self, filenames):
        """
        Load python version of Cifar100 dataset.
        """
        # Load the first batch of data to get shape info.
        filename = filenames[0]
        with open(filename, "rb") as f:
            tmp = pickle.load(f)
            data = tmp["data"]
            labels = np.array(tmp["fine_labels"])

        # Load the rest.
        for filename in filenames[1:]:
            with open(filename, "rb") as f:
                tmp = pickle.load(f)
                data = np.append(data, tmp["data"], 1)
                labels = np.append(labels, tmp["fine_labels"])

        data = np.reshape(data, [-1, 3, 32, 32])
        data = np.einsum("nchw->nhwc", data)

        return DataSet(data, labels)

    def _read(self):
        """
        Construct input for CIFAR100 evaluation using the Reader ops.
        """
        self._maybe_convert_to_tf()
        self._read_from_tfrecord()

    def _read_from_tfrecord(self):
        # Read and set up data tensors.
        filename = os.path.join(self.work_dir, 'cifar100_training.tfrecords')
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer([filename])
        self._training_datum, self._training_label \
            = self._get_sample_tensors_from_tfrecords(filename_queue)

        filename = os.path.join(self.work_dir, 'cifar100_test.tfrecords')
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
        TRAINING_TF_FILENAME = "cifar100_training"
        if not os.path.exists(
                os.path.join(self.work_dir,
                             TRAINING_TF_FILENAME + ".tfrecords")):
            # Read the numpy data in and convert it to TFRecord.
            imgs = np.load(os.path.join(self.work_dir,
                                        "pylearn2_gcn_whitened",
                                        "train.npy"))
            imgs = imgs.reshape([Cifar100TFSource.SAMPLE_NUM, 3, 32, 32])
            imgs = np.einsum("nchw->nhwc", imgs)

            # Load training set into memory.
            train_filenames = [os.path.join(self.work_dir,
                                            'cifar-100-python',
                                            "train")]
            training_labels = self._load_cifar100_python(
                train_filenames).labels
            self._convert_to_tf(imgs, training_labels, TRAINING_TF_FILENAME)

        TEST_TF_FILENAME = "cifar100_test"
        if not os.path.exists(
                os.path.join(self.work_dir, TEST_TF_FILENAME + ".tfrecords")):
            imgs = np.load(os.path.join(self.work_dir,
                                        "pylearn2_gcn_whitened",
                                        "test.npy"))
            imgs = imgs.reshape([10000, 3, 32, 32])
            imgs = np.einsum("nchw->nhwc", imgs)
            test_filenames = [os.path.join(self.work_dir,
                                           'cifar-100-python',
                                           'test')]
            test_labels = self._load_cifar100_python(test_filenames).labels
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
                    [32, 32, 3],
                    tf.float32),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['label'], tf.int32)
        image = features["image_raw"]

        return image, label
