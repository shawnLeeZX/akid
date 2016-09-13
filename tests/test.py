"""
This module offer a top level class for testing.
"""
import unittest
from unittest import TestCase

from akid import AKID_DATA_PATH
from akid import (
    MNISTFeedSource,
    Cifar10TFSource,
    Kid,
    FeedSensor,
    MomentumKongFu
)
from akid.models.brains import OneLayerBrain


def main():
    unittest.main()


class AKidTestCase(TestCase):
    def setUp(self):
        pass


class TestFactory(object):
    """
    A top level class for testing `akid`.
    """

    @staticmethod
    def get_test_brain(using_moving_average=False):
        """
        Args:
            using_moving_average: A Boolean. Using moving average to do test or
                not for the test brain.
        """
        if using_moving_average:
            return OneLayerBrain(moving_average_decay=0.9, name="test_brain")
        else:
            return OneLayerBrain(name="test_brain")

    @staticmethod
    def get_test_feed_source():
        return MNISTFeedSource(
            name="MNIST_feed",
            url='http://yann.lecun.com/exdb/mnist/',
            num_train=50000,
            num_val=5000,
            scale=True)

    @staticmethod
    def get_test_tf_source():
        return Cifar10TFSource(
            name="CIFAR10",
            url='http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
            work_dir=AKID_DATA_PATH + '/cifar10',
            num_train=50000,
            num_val=10000)

    @staticmethod
    def get_test_kid(source, brain):
        """
        Return a default kid given a source and a brain.
        """
        return Kid(
            FeedSensor(source_in=source, name='data'),
            brain,
            MomentumKongFu(),
            max_steps=900)
