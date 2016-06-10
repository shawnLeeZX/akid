import os
import sys
import tarfile
import urllib
import cPickle as pickle

import numpy as np
import tensorflow as tf

from ..core.sources import InMemoryFeedSouce, SupervisedSource, TFSource
from .datasets import DataSet, DataSets


class Cifar10FeedSource(InMemoryFeedSouce, SupervisedSource):
    """
    A concrete `FeedSource` for Cifar10 dataset.
    """
    def __init__(self, use_zca=False, **kwargs):
        """
        Args:
            use_zca: Boolean
                Use ZCA whitened data or not. If this is specified, the ZCA
                whitened data has to be in `work_dir` already.
        """
        super(Cifar10FeedSource, self).__init__(**kwargs)
        self.use_zca = use_zca

    @property
    def shape(self):
        return [32, 32, 3]

    @property
    def label_shape(self):
        return [1]

    def _load(self):
        self._maybe_download_and_extract()

        # Load training set into memory.
        train_filenames = [os.path.join(self.work_dir, 'cifar-10-batches-py',
                                        'data_batch_%d' % i)
                           for i in xrange(1, 6)]
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
            training_dataset = DataSet(imgs, training_dataset.labels)

            imgs = np.load(os.path.join(self.work_dir,
                                        "pylearn2_gcn_whitened",
                                        "test.npy"))
            imgs = imgs.reshape([10000, 3, 32, 32])
            imgs = np.einsum("nchw->nhwc", imgs)
            test_dataset = DataSet(imgs, test_dataset.labels)

        return DataSets(training_dataset, test_dataset)

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

    def _maybe_download_and_extract(self):
        """Download and extract the tarball from Alex's website."""
        # TODO(Shuai) this download piece does not work at all here. It is
        # copied from the tf tutorial directly and only checks the existence of
        # binary version of cifar10.
        dest_directory = self.work_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
            filename = self.url.split('/')[-1]
            filepath = os.path.join(dest_directory, filename)
            if not os.path.exists(filepath):
                def _progress(count, block_size, total_size):
                    sys.stdout.write(
                        '\r>> Downloading %s %.1f%%' %
                        (filename,
                         float(count * block_size) /
                         float(total_size) * 100.0))
                    sys.stdout.flush()
                filepath, _ = urllib.urlretrieve(
                    self.url, filepath, reporthook=_progress)
                print()
                statinfo = os.stat(filepath)
                print('Succesfully downloaded',
                      filename, statinfo.st_size, 'bytes.')
                tarfile.open(filepath, 'r:gz').extractall(dest_directory)


class Cifar10TFSource(TFSource, SupervisedSource):
    """
    A concrete `Source` for Cifar10 dataset.
    """
    IMAGE_SIZE = 24

    @property
    def training_datum(self):
        return self._training_datum

    @property
    def val_datum(self):
        return self._val_datum

    @property
    def training_label(self):
        return self._training_label

    @property
    def val_label(self):
        return self._val_label

    def _read(self):
        """
        Construct input for CIFAR evaluation using the Reader ops.
        """
        self._maybe_download_and_extract()

        filenames = [os.path.join(self.work_dir, 'cifar-10-batches-bin',
                                  'data_batch_%d.bin' % i)
                     for i in xrange(1, 6)]
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

    def _maybe_download_and_extract(self):
        """Download and extract the tarball from Alex's website."""
        dest_directory = self.work_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = self.url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    '\r>> Downloading %s %.1f%%' %
                    (filename,
                     float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.urlretrieve(
                self.url, filepath, reporthook=_progress)
            print()
            statinfo = os.stat(filepath)
            print('Succesfully downloaded',
                  filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)
