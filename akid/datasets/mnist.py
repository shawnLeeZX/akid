from __future__ import absolute_import
from __future__ import print_function
import os
import six.moves.urllib.request, six.moves.urllib.parse, six.moves.urllib.error
import gzip
import zipfile

import numpy as np

from torchvision import datasets, transforms

from ..core.sources import Source, InMemoryFeedSource, SupervisedSource, StaticSource
from .datasets import DataSet, DataSets
from .. import backend as A


class MNISTFeedSource(InMemoryFeedSource, SupervisedSource):
    """
    A concrete `Source` for MNIST dataset.
    """
    @property
    def shape(self):
        return [28, 28, 1]

    @property
    def label_shape(self):
        return [1]

    def _maybe_download(self, filename, work_directory):
        """Download the data from Yann's website, unless it's already here."""
        if not os.path.exists(work_directory):
            os.mkdir(work_directory)
        filepath = os.path.join(work_directory, filename)
        if not os.path.exists(filepath):
            filepath, _ = six.moves.urllib.request.urlretrieve(self.url + filename, filepath)
            statinfo = os.stat(filepath)
            self.log('Successfully downloaded {} {} bytes.'.format(filename, statinfo.st_size))
        return filepath

    def _load(self, fake_data=False, one_hot=False):
        if fake_data:
            training_dataset = DataSet([], [], fake_data=True)
            validation_dataset = DataSet([], [], fake_data=True)
            test_dataset = DataSet([], [], fake_data=True)
            return DataSets(training_dataset, validation_dataset, test_dataset)

        TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
        TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
        TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
        TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

        local_file = self._maybe_download(TRAIN_IMAGES, self.work_dir)
        train_images = self._extract_images(local_file)

        local_file = self._maybe_download(TRAIN_LABELS, self.work_dir)
        train_labels = self._extract_labels(local_file, one_hot=one_hot)

        local_file = self._maybe_download(TEST_IMAGES, self.work_dir)
        test_images = self._extract_images(local_file)

        local_file = self._maybe_download(TEST_LABELS, self.work_dir)
        test_labels = self._extract_labels(local_file, one_hot=one_hot)

        VALIDATION_SIZE = self.validation_rate * self.num_train
        validation_images = train_images[:VALIDATION_SIZE]
        validation_labels = train_labels[:VALIDATION_SIZE]
        train_images = train_images[VALIDATION_SIZE:]
        train_labels = train_labels[VALIDATION_SIZE:]

        training_dataset = DataSet(train_images,
                                   train_labels,
                                   center=self.center,
                                   scale=self.scale)
        validation_dataset = DataSet(validation_images,
                                     validation_labels,
                                     center=self.center,
                                     scale=self.scale)
        test_dataset = DataSet(test_images,
                               test_labels,
                               shuffle=False,
                               center=self.center,
                               scale=self.scale)

        return DataSets(training_dataset, test_dataset, validation_dataset)

    def _read32(self, bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return int(np.frombuffer(bytestream.read(4), dtype=dt))

    def _dense_to_one_hot(self, labels_dense, num_classes=10):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def _extract_images(self, filename):
        """
        Extract the images into a 4D uint8 np array [index, y, x, depth].
        """
        self.log('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2051:
                raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
            num_images = self._read32(bytestream)
            rows = self._read32(bytestream)
            cols = self._read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols, 1)
            return data

    def _extract_labels(self, filename, one_hot=False):
        """Extract the labels into a 1D uint8 numpy array [index]."""
        print(('Extracting', filename))
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2049:
                raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
            num_items = self._read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            if one_hot:
                return self._dense_to_one_hot(labels)
            return labels


class RotatedMNISTFeedSource(InMemoryFeedSource, SupervisedSource):
    """
    A concrete `Source` for rotated MNIST dataset.
    """
    @property
    def shape(self):
        return [28, 28, 1]

    @property
    def label_shape(self):
        return [1]

    def _maybe_download_and_extract(self, filename):
        """
        Download the data from Yann's website, unless it's already here.

        File contents will be extracted in the `self.work_dir`.
        """
        if not os.path.exists(self.work_dir):
            os.mkdir(self.work_dir)
        filepath = os.path.join(self.work_dir, filename)
        if not os.path.exists(filepath):
            filepath, _ = six.moves.urllib.request.urlretrieve(self.url + filename, filepath)
            statinfo = os.stat(filepath)
            self.log('Successfully downloaded', filename, statinfo.st_size,
                     'bytes.')
            self.log('Extracting zip file ... ')
            f = zipfile.ZipFile(filepath)
            f.extractall(path=self.work_dir)
            self.log('Extraction finished ... ')

    def _load(self, fake_data=False, one_hot=False):
        filename = self.url.split('/')[-1]
        filepath = os.path.join(self.work_dir, filename)
        if not os.path.exists(filepath):
            self._maybe_download_and_extract(filename)

        TRAIN_DATA_PATH = os.path.join(
            self.work_dir,
            'mnist_all_rotation_normalized_float_train_valid.amat')
        TEST_DATA_PATH = os.path.join(
            self.work_dir,
            'mnist_all_rotation_normalized_float_test.amat')

        train_data = np.loadtxt(TRAIN_DATA_PATH)
        test_data = np.loadtxt(TEST_DATA_PATH)
        training_images = np.reshape(train_data[:self.num_train, 0:-1],
                                     [-1, 28, 28, 1])
        training_labels = train_data[:self.num_train, -1]
        test_images = np.reshape(test_data[:self.num_val, 0:-1],
                                 [-1, 28, 28, 1])
        test_labels = test_data[:self.num_val, -1]

        training_dataset = DataSet(training_images,
                                   training_labels,
                                   center=self.center,
                                   scale=self.scale)
        test_dataset = DataSet(test_images,
                               test_labels,
                               center=self.center,
                               scale=self.scale)

        return DataSets(training_dataset, test_dataset)


class MNISTTorchSource(StaticSource, SupervisedSource):
    def _setup(self):
        self.dataset = datasets.MNIST(self.work_dir, train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))]))
        self.val_dataset = datasets.MNIST(self.work_dir, train=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))]))
    def _forward(self):
        pass


class MNISTSource(Source):
    def _setup(self):
        if not hasattr(self, "_data"):
            self._data = datasets.MNIST(self.work_dir, train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))]))

    def set_mode(self, mode):
        super(MNISTSource, self).set_mode(mode)
        if mode == A.Mode.TRAIN:
            self._data = datasets.MNIST(self.work_dir, train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))]))
        elif mode == A.Mode.VAL:
            self._data = datasets.MNIST(self.work_dir, train=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))]))
        else:
            raise ValueError("Mode {} not supported yet.".format(mode))

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return len(self._data)
