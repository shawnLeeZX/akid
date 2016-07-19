import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from akid import AKID_DATA_PATH
from akid.tests.test import TestCase, main
from akid.datasets import (
    Cifar10FeedSource,
    MNISTFeedSource,
    RotatedMNISTFeedSource
)


class TestSource(TestCase):
    def test_mnist_feed_source(self):
        source = MNISTFeedSource(
            name="MNIST_feed",
            url='http://yann.lecun.com/exdb/mnist/',
            num_train=50000,
            num_val=5000,
            scale=True)

        source.setup()
        imgs, labels = source.get_batch(1, True)
        img = np.squeeze(imgs)
        plt.imshow(img)
        print("The class label is {}.".format(labels[0]))
        plt.show()

        imgs, labels = source.get_batch(1, False)
        img = np.squeeze(imgs)
        plt.imshow(img)
        print("The class label is {}.".format(labels[0]))
        plt.show()

    def test_rotated_mnist_feed_source(self):
        source = RotatedMNISTFeedSource(
            name="Rotated_MNIST_feed",
            url='http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip',
            work_dir=AKID_DATA_PATH + '/rotated_mnist',
            num_train=12000,
            num_val=50000,
            scale=True)

        source.setup()
        imgs, labels = source.get_batch(1, True)
        img = np.squeeze(imgs)
        plt.imshow(img)
        print("The class label is {}.".format(labels[0]))
        plt.show()

        imgs, labels = source.get_batch(1, False)
        img = np.squeeze(imgs)
        plt.imshow(img)
        print("The class label is {}.".format(labels[0]))
        plt.show()

    def test_cifar_feed_source(self):
        # Just read a batch and plot channels of images to see.
        source = Cifar10FeedSource(
            name="CIFAR10",
            url='http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
            work_dir=AKID_DATA_PATH + '/cifar10',
            num_train=50000,
            num_val=10000)

        source.setup()
        imgs, labels = source.get_batch(1, True)
        plt.imshow(imgs[0, ...])
        print("The class label is {}.".format(labels[0]))
        plt.show()

        imgs, labels = source.get_batch(1, False)
        plt.imshow(imgs[0, ...])
        print("The class label is {}.".format(labels[0]))
        plt.show()

    def test_cifar_zca_feed_source(self):
        # Just read a batch and plot channels of images to see.
        source = Cifar10FeedSource(
            name="CIFAR10",
            url='http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
            work_dir=AKID_DATA_PATH + '/cifar10',
            use_zca=True,
            num_train=50000,
            num_val=10000)

        source.setup()
        imgs, labels = source.get_batch(1, True)
        for c in xrange(0, 3):
            plt.imshow(imgs[0, ..., c])
            print("The class label is {}.".format(labels[0]))
            plt.show()
        imgs, labels = source.get_batch(1, False)
        for c in xrange(0, 3):
            plt.imshow(imgs[0, ..., c])
            print("The class label is {}.".format(labels[0]))
            plt.show()


if __name__ == "__main__":
    main()
