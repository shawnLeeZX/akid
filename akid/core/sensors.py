"""
The interface between nature and an (artificial) signal processing system,
saying a brain, is a `Sensor`. It does data augmentation (if needed), and
batches datum from `Source`.

Strictly speaking, the functional role of a sensor is to convert the signal in
the natural form to a form the data processing engine, which is the brain in
this case, could process. It is a Analog/Digital converter. However, the input
from `Source` is already in digital form, so this function is not there
anymore. But the data batching, augmentation and so on could still be put in
preprocessing. Thus we still use the name sensor for concept reuse.

Mathematically, it is a system made up with a series of linked blocks that do
data augmentation.

As an example, again saying in supervised setting, a sensor is a block that
takes a data source and output sensed (batched and augmented) data. A sensor
needs to be used along with a source. A concrete example could be the sensor
for the MNIST dataset. Taking a `Source`, we could make a sensor::

      sensor = FeedSensor(name='data', source_in=source)

The type of a sensor must match that of a source.
"""
from __future__ import absolute_import, division, print_function

import sys
import abc
import inspect

import tensorflow as tf

from .jokers import JokerSystem
from .blocks import Block
from ..utils import glog as log
from . import sources
from .common import TRAIN_SUMMARY_COLLECTION, VALID_SUMMARY_COLLECTION


class Sensor(Block):
    """
    The top level abstract sensor to preprocessing raw data received from
    `Source`, such as batching, data augmentation etc.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 source_in,
                 batch_size=100,
                 val_batch_size=100,
                 **kwargs):
        """
        Args:
            name: str
                Name of this sensor.
            source_in: Source
                Where data should be sensed from.
            batch_size: int
                The number of samples a time the sensor would provide.
            val_batch_size: int
                The number of samples a time the sensor would provide when
                doing validation. It is supposed to evenly divide the number of
                validation samples.
        """
        super(Sensor, self).__init__(self, **kwargs)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.source = source_in

    def data(self, get_val=False):
        """
        Args:
            get_val: A Boolean. If True, return validation data, otherwise,
                return training data.
        """
        if get_val:
            return self.val_data
        else:
            return self.training_data

    def labels(self, get_val=False):
        """
        Args:
            get_val: A Boolean. If True, return validation labels, otherwise,
                return training labels.
        """
        if issubclass(type(self.source), sources.SupervisedSource):
            if get_val:
                return self.val_labels
            else:
                return self.training_labels
        else:
            log.error("Sensor {} is not supposed to provide"
                      "labels".format(self.name))

        sys.exit()

    @abc.abstractmethod
    def _setup_training_data(self):
        """
        An abstract method each subclass must implement to provide training
        data.

        Returns:
            A tuple of tensors, (training_data, training_labels), should be
            returned if the source of this sensor is a supervised one;
            otherwise, a tensor of training data should be returned.
        """
        raise NotImplementedError("Each sensor needs to implement the method"
                                  " to actually provide data as tensors.")
        sys.exit()

    @abc.abstractmethod
    def _setup_val_data(self):
        """
        An abstract method each subclass must implement to provide validation
        data.

        Returns:
            A tuple of tensors, (training_data, training_labels), should be
            returned if the source of this sensor is a supervised one;
            otherwise, a tensor of training data should be returned.
        """
        raise NotImplementedError("Each sensor needs to implement the method"
                                  " to actually provide data as tensors.")
        sys.exit()

    def _pre_setup(self):
        if issubclass(type(self.source), sources.StaticSource):
            self.num_batches_per_epoch_train \
                = (self.source.num_train - 1) // self.batch_size + 1
            self.num_batches_per_epoch_val \
                = (self.source.num_val - 1) // self.val_batch_size + 1
            log.info("A epoch of training set contains {} batches".format(
                self.num_batches_per_epoch_train))
            log.info("A epoch of validation set contains {} batches".format(
                self.num_batches_per_epoch_val))

    def _post_setup_shared(self):
        if self.do_summary:
            self._image_summary(self.training_data.op.name,
                                self.training_data,
                                TRAIN_SUMMARY_COLLECTION)
            self._image_summary(self.val_data.op.name,
                                self.val_data,
                                VALID_SUMMARY_COLLECTION)

    def _image_summary(self, name, image_batch, collection):
            tf.histogram_summary(name,
                                 image_batch,
                                 collections=[collection])
            tf.image_summary(name,
                             image_batch,
                             collections=[collection])

    def _setup(self):
        """
        Generate placeholder or tensor variables to represent the the input
        data.
        """
        self.source.setup()

        log.info("Setting up training sensor ... ")
        if issubclass(type(self.source), sources.SupervisedSource):
            self.training_data, self.training_labels \
                = self._setup_training_data()
        else:
            self.training_data = self._setup_training_data()
        log.info("Setting up val sensor ... ")
        if issubclass(type(self.source), sources.SupervisedSource):
            self.val_data, self.val_labels = self._setup_val_data()
        else:
            self.val_data = self._setup_val_data()

        log.info("Finished setting up sensor.")


class ShuffleQueueSensor(Sensor):
    """
    A `Sensor` that holds a shuffle queue, which would pre-load
    `min_fraction_of_examples_in_queue` number of examples and sample batches
    from the queue. Preloading could reduce disk latency and doing data
    augmentation ahead of time, and load a relatively large number of samples
    could ensure that the random shuffling has good
    mixing properties.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, min_fraction_of_examples_in_queue=0.1, **kwargs):
        """
        Args:
            min_fraction_of_examples_in_queue: A fraction less than 1.
        """
        super(ShuffleQueueSensor, self).__init__(**kwargs)
        assert min_fraction_of_examples_in_queue > 0 and \
            min_fraction_of_examples_in_queue <= 1, \
            "min_fraction_of_examples_in_queue should be between 0 and 1"
        self.min_fraction_of_examples_in_queue \
            = min_fraction_of_examples_in_queue


class IntegratedSensor(ShuffleQueueSensor):
    """
    A concrete `Sensor` uses Reader Op of tensorflow to read data directly as a
    tensor into a computational graph instead of being a placeholder as in
    `FeedSensor`.

    Optionally, it could also do data augmentation. It holds two
    `LinkedSystem`s, `training_jokers` and `val_jokers`, which do data
    processing on training datum and validation datum respectively.
    """
    def __init__(self, **kwargs):
        super(IntegratedSensor, self).__init__(**kwargs)

        # Keep two LinkedSystem to hold Jokers that may apply to training and
        # validation data.
        self.training_jokers = JokerSystem(name="training_joker")
        self.val_jokers = JokerSystem(name="val_joker")

    def _setup_training_data(self):
        # TODO(Shuai): Handle the case where the source has no labels.
        self.training_jokers.setup(self.source.training_datum)
        augmented_training_datum = self.training_jokers.data
        min_queue_examples = int(self.source.num_train *
                                 self.min_fraction_of_examples_in_queue)

        batch_list = self._generate_image_and_label_batch(
            self.batch_size,
            augmented_training_datum,
            self.source.training_label,
            min_queue_examples,
            "train_data")
        training_data = batch_list[0]
        training_labels = batch_list[1:]

        return training_data, training_labels

    def _setup_val_data(self):
        # TODO(Shuai): Handle the case where the source has no labels.
        self.val_jokers.setup(self.source.val_datum)
        processed_val_datum = self.val_jokers.data
        min_queue_examples = int(self.source.num_val *
                                 self.min_fraction_of_examples_in_queue)

        batch_list = self._generate_image_and_label_batch(
            self.val_batch_size,
            processed_val_datum,
            self.source.val_label,
            min_queue_examples,
            "val_data")
        val_data = batch_list[0]
        val_labels = batch_list[1:]

        return val_data, val_labels

    def attach(self, joker, to_val=False):
        """
        Attach a joker to a joker system. If `to_val` is True, attach to
        training joker system, otherwise to validation joker system.
        """
        if to_val:
            self.val_jokers.attach(joker)
        else:
            self.training_jokers.attach(joker)

    def _post_setup_shared(self):
        super(IntegratedSensor, self)._post_setup_shared()
        if self.do_summary:
            # Do image summary on raw images if we have done data augmentation.
            if not self.training_jokers.is_empty:
                self._raw_datum_summary(self.training_data.op.name + "_raw",
                                        self.source.training_datum,
                                        TRAIN_SUMMARY_COLLECTION)
            if not self.val_jokers.is_empty:
                self._raw_datum_summary(self.val_data.op.name + "_raw",
                                        self.source.val_datum,
                                        VALID_SUMMARY_COLLECTION)

    def _raw_datum_summary(self, name, datum, collection):
        # Since image summary only takes image batches, we package each
        # image into a batch.
        shape = datum.get_shape().as_list()
        shape.insert(0, 1)
        datum_batch = tf.reshape(datum, shape)
        tf.image_summary(name,
                         datum_batch,
                         collections=[collection])

    def _generate_image_and_label_batch(
            self, batch_size, image, label, min_queue_examples, name):
        """Construct a queued batch of images and labels.

        Args:
            batch_size: An integer.
            image: 3-D Tensor of [IMAGE_SIZE, IMAGE_SIZE, 3] of type.float32.
            label: 1-D Tensor of type.int32 or a list of them.
            min_queue_examples: int32, minimum number of samples to retain
            in the queue that provides of batches of examples.

        Returns:
            batch_list: a list
                A list of batched tensors of passed in `image` and `label`. The
                order how they are passed in is preserved in the list.
        """
        # Having errors when thread number is too high.
        # num_preprocess_threads = 16
        num_preprocess_threads = 4
        input_list = [image]
        input_list.extend(label) if type(label) is list \
            else input_list.append(label)
        batch_list = tf.train.shuffle_batch(
            input_list,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples,
            name=name)

        for i, b in enumerate(batch_list):
            batch_list[i] = tf.squeeze(b)

        return batch_list


class FeedSensor(Sensor):
    """
    Sense from a `FeedSource` to supply data to a `Kid`.
    """
    def _setup_training_data(self):
        return self._make_placeholder("train_data", self.batch_size)

    def _setup_val_data(self):
        return self._make_placeholder("val_data", self.val_batch_size)

    def _make_placeholder(self, name, batch_size):
        data_shape = self.source.shape
        data_shape.insert(0, batch_size)
        data = tf.placeholder(tf.float32, shape=data_shape, name=name)

        if issubclass(type(self.source), sources.SupervisedSource):
            label_shape = self.source.label_shape
            if len(label_shape) is 1:
                label_shape = [batch_size]
            else:
                label_shape.insert(0, batch_size)

            labels = tf.placeholder(tf.int32, shape=label_shape, name=name)

            return data, labels

        return data

    def fill_feed_dict(self, get_val=False):
        """Supply a batch of training examples in form of feed dict.

        A feed_dict takes the form of:
        feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
            ....
        }

        Args:
            get_val: A Boolean. If True, return validation data, otherwise,
                return training data.

        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        # Create the feed_dict for the placeholders filled with the next
        # `batch size ` examples.
        batch_size = self.val_batch_size if get_val else self.batch_size
        images_feed, labels_feed = self.source.get_batch(batch_size,
                                                         get_val)
        feed_dict = {
            self.data(get_val): images_feed,
            self.labels(get_val): labels_feed,
        }
        return feed_dict


__all__ = [name for name, x in locals().items() if
           not inspect.ismodule(x) and not inspect.isabstract(x)]
