import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from akid import AKID_DATA_PATH
from akid.utils.test import AKidTestCase, main
from akid.datasets import Cifar10FeedSource, Cifar10TFSource
from akid.datasets import Cifar100TFSource
from akid.datasets import MNISTFeedSource, RotatedMNISTFeedSource
from akid import LearningRateScheme


class TestSource(AKidTestCase):
    def test_mnist_feed_source(self):
        source = MNISTFeedSource(
            name="MNIST_feed",
            url='http://yann.lecun.com/exdb/mnist/',
            num_train=50000,
            num_val=5000,
            scale=True)

        source.forward()
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

        source.forward()
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

    def test_cifar10_feed_source(self):
        # Just read a batch and plot channels of images to see.
        source = Cifar10FeedSource(
            name="CIFAR10",
            url='http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
            work_dir=AKID_DATA_PATH + '/cifar10',
            num_train=50000,
            num_val=10000)

        source.forward()
        imgs, labels = source.get_batch(1, True)
        plt.imshow(imgs[0, ...])
        print("The class label is {}.".format(labels[0]))
        plt.show()

        imgs, labels = source.get_batch(1, False)
        plt.imshow(imgs[0, ...])
        print("The class label is {}.".format(labels[0]))
        plt.show()

    def test_cifar10_zca_feed_source(self):
        # Just read a batch and plot channels of images to see.
        source = Cifar10FeedSource(
            name="CIFAR10",
            url='http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
            work_dir=AKID_DATA_PATH + '/cifar10',
            use_zca=True,
            num_train=50000,
            num_val=10000)

        source.forward()
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

    def test_cifar10_zca_tf_source(self):
        from akid.models.brains import VGGNet
        from akid import IntegratedSensor, Kid, GradientDescentKongFu
        source = Cifar10TFSource(
            name="CIFAR10",
            url='http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
            work_dir=AKID_DATA_PATH + '/cifar10',
            use_zca=True,
            num_train=50000,
            num_val=10000)

        sensor = IntegratedSensor(source_in=source,
                                  batch_size=128,
                                  val_batch_size=100,
                                  name='data')

        brain = VGGNet(padding="SAME", name="VGGNet")
        kid = Kid(
            sensor,
            brain,
            GradientDescentKongFu(
                lr_scheme={
                    "name": LearningRateScheme.exp_decay,
                    "base_lr": 0.1,
                    "decay_rate": 0.1,
                    "num_batches_per_epoch": 391,
                    "decay_epoch_num": 350}),
            summary_on_val=True,
            max_steps=1000)
        kid.setup()

        loss = kid.practice()
        assert loss < 1.6

    def test_cifar100_zca_tf_source(self):
        from akid.models.brains import VGGNet
        from akid import IntegratedSensor, Kid, GradientDescentKongFu
        source = Cifar100TFSource(
            name="CIFAR100",
            url='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
            work_dir=AKID_DATA_PATH + '/cifar100',
            num_train=50000,
            num_val=10000)

        sensor = IntegratedSensor(source_in=source,
                                  batch_size=128,
                                  val_batch_size=100,
                                  name='data')

        brain = VGGNet(class_num=100, name="VGGNet")
        kid = Kid(
            sensor,
            brain,
            GradientDescentKongFu(
                lr_scheme={
                    "name": LearningRateScheme.exp_decay,
                    "base_lr": 0.1,
                    "decay_rate": 0.1,
                    "num_batches_per_epoch": 391,
                    "decay_epoch_num": 350}),
            summary_on_val=True,
            max_steps=1000)
        kid.setup()

        loss = kid.practice()
        assert loss < 3.7

    def test_hcifar100_source(self):
        from akid import HCifar100TFSource
        from akid.models.brains import VGGNet
        from akid import IntegratedSensor, Kid, GradientDescentKongFu
        from akid.layers import (
            GroupSoftmaxWithLossLayer,
            CollapseOutLayer,
            SoftmaxWithLossLayer
        )
        source = HCifar100TFSource(
            name="CIFAR100",
            url='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
            work_dir=AKID_DATA_PATH + '/cifar100',
            num_train=50000,
            num_val=10000)
        source.forward()

        sensor = IntegratedSensor(source_in=source,
                                  batch_size=128,
                                  val_batch_size=100,
                                  name='data')

        brain = VGGNet(class_num=100,
                       loss_layer=(GroupSoftmaxWithLossLayer,
                                   {"group_size": 5,
                                    "inputs": [
                                        {"name": "ip2", "idxs": [0]},
                                        {"name": "system_in", "idxs": [2, 3]}]
                                    }),
                       name="VGGNet")
        brain.attach(CollapseOutLayer(group_size=5, name="maxout"))
        brain.attach(SoftmaxWithLossLayer(
            class_num=20,
            inputs=[
                {"name": "maxout", "idxs": [0]},
                {"name": "system_in", "idxs": [1]}],
            name="super_class_loss"))
        kid = Kid(
            sensor,
            brain,
            GradientDescentKongFu(
                lr_scheme={
                    "name": LearningRateScheme.exp_decay,
                    "base_lr": 0.1,
                    "decay_rate": 0.1,
                    "num_batches_per_epoch": 391,
                    "decay_epoch_num": 350}),
            summary_on_val=True,
            max_steps=2000)
        kid.setup()

        loss = kid.practice()
        assert loss < 7.3

    def test_imagenet_source(self):
        # Ideally, this source is supposed to test against the read image and
        # labels, however, for time's sake ... just check compilation and
        # runtime errors for now.
        from akid import ImagenetTFSource
        source = ImagenetTFSource(
            name="Imagenet",
            url=None,
            work_dir=AKID_DATA_PATH + "/small_imagenet",
            num_train=84321,
            num_val=3300)
        source.forward()

        with tf.Session() as sess:
            sess.run([source.training_datum,
                      source.training_label,
                      source.val_datum,
                      source.val_label])

        # Again just test for compilation and runtime error.
        source = ImagenetTFSource(
            has_super_label=False,
            name="Imagenet",
            url=None,
            work_dir=AKID_DATA_PATH + "/imagenet",
            num_train=1281167,
            num_val=50000)
        source.forward()

        with tf.Session() as sess:
            sess.run([source.training_datum,
                      source.training_label,
                      source.val_datum,
                      source.val_label])

if __name__ == "__main__":
    main()
