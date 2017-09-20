"""
This module offer a top level class for testing.
"""
import sys
import os
import pdb
import functools
import traceback

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
    def assertNdarrayEquals(self, a, b):
        if not (a == b).all():
            msg = self._formatMessage(None, '\n{}\n == \n{}\n'.format(a, b))
            raise self.failureException(msg)


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
            debug=True,
            max_steps=900)


class TestSuite():
    def suite(self): #Function stores all the modules to be tested
        modules_to_test = []
        test_dir = os.listdir('.')
        modules_to_test.append('test_systems')
        modules_to_test.append('test_loss_layers')
        modules_to_test.append('test_brain')
        modules_to_test.append('test_synapse_layers')
        modules_to_test.append('test_kongfus')
        # for test in test_dir:
        #     if test.startswith('test') and test.endswith('.py'):
        #         modules_to_test.append(test.rstrip('.py'))

        all_tests = unittest.TestSuite()
        for module in map(__import__, modules_to_test):
            all_tests.addTest(unittest.findTestCases(module))
        return all_tests


def debug_on(*exceptions):
    if not exceptions:
        exceptions = (AssertionError, )
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exceptions:
                info = sys.exc_info()
                traceback.print_exception(*info)
                pdb.post_mortem(info[2])
        return wrapper
    return decorator
