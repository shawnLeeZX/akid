"""
This module holds classes to model datasets.
"""
from __future__ import absolute_import
import numpy
from six.moves import range


PIXEL_DEPTH = 255


class DataSet(object):

    def __init__(self,
                 images,
                 labels,
                 center=False,
                 scale=False,
                 shuffle=True,
                 fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]
            if center:
                images = images - (PIXEL_DEPTH / 2.0)
            if scale:
                images = images.astype(numpy.float32)
                images = images / PIXEL_DEPTH

        self._images = images
        self._labels = labels
        self.shuffle = shuffle
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in range(784)]
            fake_label = 0
            return [fake_image for _ in range(batch_size)], [
                fake_label for _ in range(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self.shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


class DataSets(object):
    """
    An class to model common aggregation of datasets.
    """
    def __init__(self, training, test=None, validation=None):
        """
        Create an aggregation of datasets.

        Args:
            training, test, validation: DataSet
        """
        self.training = training
        self.test = test
        self.validation = validation
