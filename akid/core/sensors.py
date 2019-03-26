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

Sensors can also include the support for functions such as multi-threaded data
prefetch to build better performed data preparation pipeline.

Mathematically, it is a system made up with a series of linked blocks that do
data augmentation.

As an example, again saying in supervised setting, a sensor is a block that
takes a data source and output sensed (batched and augmented) data. A sensor
needs to be used along with a source. A concrete example could be the sensor
for the MNIST dataset. Taking a `Source`, we could make a sensor::

      sensor = SimpleSensor(name='data', source_in=source)

Similar with `Source`, a sensor has multiple modes to supply data. Refer to the
documentation of `Source` for more details.
"""
from __future__ import absolute_import, division, print_function

import sys
import abc
import inspect
from deprecated import deprecated
import six
from six.moves import range

if sys.version_info[0] == 2:
    from six.moves.queue import Queue
    import six.moves.queue as queue
else:
    from queue import Queue
    import queue
import threading

import tensorflow as tf
import torch as th

from .jokers import JokerSystem
from .blocks import ProcessingBlock
from . import sources
from . import samplers
from .common import TRAIN_SUMMARY_COLLECTION, VALID_SUMMARY_COLLECTION
from .events import DataPrefetchThreadsDeadEvent
from akid import backend as A

import sys
import os
import pdb
import functools
import traceback


def _data_fetching_worker(source, index_queue, data_queue, done_event):
    # If index queue is not empty, then fetch the indices of a batch, and load
    # the data to data queue.
    while True:
        # If the main thread is dead, quit.
        for t in threading.enumerate():
            if t.name == "MainThread":
                if not t.is_alive():
                    return
        # If the done event is set, quit.
        if done_event.is_set():
            return

        try:
            indices = index_queue.get(timeout=A.TIMEOUT)
            data = source.get(indices)
            data_queue.put(data)
        except queue.Empty:
            continue


class Sensor(ProcessingBlock):
    """
    The top level abstract sensor to prepare data.

    The main thread would put indices to prefetch in an index queue. A data
    queue should be implemented by any subclass to store the data prefetched.

    By the default, a sensor is in the "train" mode. To use it in other modes,
    one needs to change the mode first, then set the batch size in that mode.
    """
    NAME = "Sensor"

    def __init__(self,
                 source_in=None,
                 batch_size=100,
                 val_batch_size=100,
                 test_batch_size=None,
                 queue_size=5,
                 sampler="shuffle",
                 **kwargs):
        """
        Args:
            source_in: Source
                Where data should be sensed from.
            batch_size: int
                The number of samples in a time the sensor would provide.
            queue_size: int
                The number of batches to prefetch.
            sampler: str
                The sampler that determines the sequence to process data.
            name: str
                Name of this sensor.
        """
        super(Sensor, self).__init__(**kwargs)
        self.source = source_in
        self.batch_size_dict = { A.Mode.TRAIN: batch_size,
                                 A.Mode.VAL: val_batch_size,
                                 A.Mode.TEST: val_batch_size if test_batch_size is None else test_batch_size}
        self.queue_size = queue_size
        self.sampler_name = sampler
        self.sampler = samplers.get(sampler, len(self.source.data))

        self.mode = A.Mode.TRAIN

    @property
    def batch_size(self):
        return self.batch_size_dict[self.mode]

    @property
    def num_batches_per_epoch(self):
        return (self.source.size - 1) // self.batch_size_dict[self.mode] + 1

    def set_batch_size(self, size):
        if type(size) is not int:
            raise ValueError("The batch size should be of type int. Type {} received".format(type(size)))
        else:
            self.batch_size_dict[self.mode] = size

    def set_mode(self, mode):
        A.check_mode(mode)

        # We need to tear down pre-fetching threads before changing the mode,
        # otherwise, source would change mode queue shutdown, thus leading to
        # the result that some new data are fetched to the old data queue, and
        # not used.
        if self.is_setup:
            self._teardown_data_queue()

        self.mode = mode
        self.source.set_mode(mode)
        self.sampler = samplers.get(self.sampler_name, len(self.source.data))

    def _setup(self):
        """
        Set up a FIFO queue to load data from source.
        """
        # Set up a queue.
        self.index_queue = Queue(self.queue_size)
        # Start loading data from source according to mode.
        ## Put indices to prefetch.
        for i in range(self.queue_size):
            self.index_queue.put(self.sampler.next(self.batch_size_dict[self.mode]))
        self._setup_data_queue()

    @property
    def data(self):
        return self._data

    @abc.abstractmethod
    def _setup_data_queue(self):
        """
        The procedure to set up a data queue should be implemented here.
        """
        pass

    @abc.abstractmethod
    def _teardown_data_queue(self):
        """
        The procedure to release resources allocated related to the data queue.
        """
        pass

    @abc.abstractproperty
    def data_queue(self):
        """
        Any subclass should has a queue that supports a `get` method to provide
        data.
        """
        pass

    def _forward(self, *args, **kwargs):
        if threading.active_count() == 1:
            # Only the main thread is alive. Raise exception.
            raise DataPrefetchThreadsDeadEvent

        ret = self.data_queue.get(timeout=A.TIMEOUT)
        A.cache_tensor_auto_scope(ret[0], "val_data" if self.is_val else "data")
        A.cache_tensor_auto_scope(ret[1], "val_labels" if self.is_val else "labels")
        self._data = ret

        self.index_queue.put(self.sampler.next(self.batch_size_dict[self.mode]))
        return self._data


class SimpleSensor(Sensor):
    """
    A simple sensor that uses a single thread to prefetch data.
    """
    def _setup_data_queue(self):
        # Set up a data queue.
        self._data_queue = Queue(self.queue_size)
        # Start loading data from source according to mode.
        ## Start the workers to fetch data.
        self.done_event = threading.Event()
        self.worker_thread = threading.Thread(
            target=_data_fetching_worker,
            args=(self.source, self.index_queue, self.data_queue, self.done_event))
        self.worker_thread.start()

    def _teardown_data_queue(self):
        self.done_event.set()
        self.worker_thread.join()

    @property
    def data_queue(self):
        return self._data_queue

    def _post_forward(self, *args, **kwargs):
        super(Sensor, self)._post_forward(*args, **kwargs)

        if self.done_first_pass and not self.is_val:
            return

        if not self.do_summary:
            return

        if not self.is_val:
            self._image_summary(A.get_name(self.data[0]),
                                self.data,
                                TRAIN_SUMMARY_COLLECTION)
            return

        if self.is_val and not self.done_first_pass_val and self.do_summary_on_val:
            self._image_summary(A.get_name(self.data[0]),
                                self.data,
                                VALID_SUMMARY_COLLECTION)

    def _image_summary(self, name, image_batch, collection):
            self.log("Do tensorboard image summary on outputs {} of {}".format(
                name, self.name))

            A.summary.histogram(name,
                                image_batch,
                                collections=[collection])
            A.summary.image(name,
                            image_batch,
                            collections=[collection])


@deprecated(reason="Legacy code. Use `Sensor` instead.")
class OldSensor(six.with_metaclass(abc.ABCMeta, ProcessingBlock)):
    """
    The top level abstract sensor to preprocessing raw data received from
    `Source`, such as batching, data augmentation etc.
    """

    def __init__(self,
                 source_in=None,
                 batch_size=100,
                 val_batch_size=100,
                 shuffle_train=True,
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
        super(OldSensor, self).__init__(self, **kwargs)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.source = source_in
        self.shuffle_train = shuffle_train
        self.done_first_pass_val = False

        if issubclass(type(self.source), sources.StaticSource):
            self.num_batches_per_epoch_train \
                = (self.source.num_train - 1) // self.batch_size + 1
            self.num_batches_per_epoch_val \
                = (self.source.num_val - 1) // self.val_batch_size + 1
            self.log("A epoch of training set contains {} batches".format(
                self.num_batches_per_epoch_train))
            self.log("A epoch of validation set contains {} batches".format(
                self.num_batches_per_epoch_val))

    @property
    def data(self):
        if self.is_val:
            return self.val_data
        else:
            return self.training_data

    @property
    def labels(self):
        if issubclass(type(self.source), sources.SupervisedSource):
            if self.is_val:
                return self.val_labels
            else:
                return self.training_labels
        else:
            raise Exception("OldSensor {} is not supposed to provide"
                            "labels".format(self.name))

    @abc.abstractmethod
    def _forward_train(self):
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
    def _forward_val(self):
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

    def _post_forward(self, *args, **kwargs):
        super(OldSensor, self)._post_forward(*args, **kwargs)

        if self.done_first_pass and not self.is_val:
            return

        if not self.do_summary:
            return

        if not self.is_val\
           or (self.is_val\
               and not self.done_first_pass_val):
            if not self.is_val:
                self._image_summary(A.get_name(self.training_data),
                                    self.training_data,
                                    TRAIN_SUMMARY_COLLECTION)
            else:
                self._image_summary(A.get_name(self.val_data),
                                    self.val_data,
                                    VALID_SUMMARY_COLLECTION)

                # Legacy_TODO: think refactor this. Should not be here. Perhaps sensor
                # should not be part of processing layer, and has a val copy.
                self.done_first_pass_val = True

    def _image_summary(self, name, image_batch, collection):
            self.log("Do tensorboard summary on outputs {} of {}".format(
                name, self.name))

            A.summary.histogram(name,
                                image_batch,
                                collections=[collection])
            A.summary.image(name,
                            image_batch,
                            collections=[collection])

    def _forward(self, *args, **kwargs):
        """
        Generate placeholder or tensor variables to represent the the input
        data.
        """
        self.source.forward()

        if not self.is_val:
            if not self.done_first_pass:
                self.log("Forwarding data from training sensor ... ")

            if issubclass(type(self.source), sources.SupervisedSource):
                self.training_data, self.training_labels \
                    = self._forward_train()
                return [self.training_data, self.training_labels]
            else:
                self.training_data = self._forward_train()

                return [self.training_data]

        else:
            if not self.done_first_pass_val:
                self.log("Forwarding data from val sensor ... ")

            if issubclass(type(self.source), sources.SupervisedSource):
                self.val_data, self.val_labels = self._forward_val()

                return [self.val_data, self.val_labels]
            else:
                self.val_data = self._forward_val()

                return [self.val_data]


class ShuffleQueueSensor(six.with_metaclass(abc.ABCMeta, OldSensor)):
    """
    A `OldSensor` that holds a shuffle queue, which would pre-load
    `min_fraction_of_examples_in_queue` number of examples and sample batches
    from the queue. Preloading could reduce disk latency and doing data
    augmentation ahead of time, and load a relatively large number of samples
    could ensure that the random shuffling has good
    mixing properties.
    """

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
    A concrete `OldSensor` uses Reader Op of tensorflow to read data directly as a
    tensor into a computational graph instead of being a placeholder as in
    `FeedSensor`.

    Optionally, it could also do data augmentation. It holds two
    `LinkedSystem`s, `training_jokers` and `val_jokers`, which do data
    processing on training datum and validation datum respectively.
    """
    def __init__(self, num_preprocess_threads=4,  **kwargs):
        super(IntegratedSensor, self).__init__(**kwargs)
        self.num_preprocess_threads = num_preprocess_threads

        # Keep two LinkedSystem to hold Jokers that may apply to training and
        # validation data.
        self.training_jokers = JokerSystem(name="training_joker")
        self.val_jokers = JokerSystem(name="val_joker")

    def _forward_train(self):
        # Legacy_TODO(Shuai): Handle the case where the source has no labels.
        self.training_jokers.forward(self.source.training_datum)
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

    def _forward_val(self):
        # Legacy_TODO(Shuai): Handle the case where the source has no labels.
        self.val_jokers.forward(self.source.val_datum)
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

    def _post_forward(self):
        super(IntegratedSensor, self)._post_forward()
        if self.do_summary:
            # Do image summary on raw images if we have done data augmentation.
            if not self.training_jokers.is_empty:
                self._raw_datum_summary(self.training_data.op.name + "_raw",
                                        self.source.training_datum,
                                        TRAIN_SUMMARY_COLLECTION)
        if self.do_summary_on_val and self.is_val and not self.val_jokers.is_empty:
            self._raw_datum_summary(self.val_data.op.name + "_raw",
                                    self.source.val_datum,
                                    VALID_SUMMARY_COLLECTION)

    def _raw_datum_summary(self, name, datum, collection):
        # Since image summary only takes image batches, we package each
        # image into a batch.
        tf.summary.image(name,
                         tf.expand_dims(datum, 0),
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
        input_list = [image]
        input_list.extend(label) if type(label) is list \
            else input_list.append(label)
        batch_list = tf.train.shuffle_batch(
            input_list,
            batch_size=batch_size,
            num_threads=self.num_preprocess_threads,
            capacity=min_queue_examples + 2 * self.num_preprocess_threads * batch_size,
            min_after_dequeue=min_queue_examples,
            name=name)

        for i, b in enumerate(batch_list):
            batch_list[i] = tf.squeeze(b)

        return batch_list


class FeedSensor(OldSensor):
    """
    Sense from a `FeedSource` to supply data to a `Kid`.
    """
    def _forward_train(self):
        return self._make_placeholder("train_data", self.batch_size)

    def _forward_val(self):
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

    def fill_feed_dict(self):
        """Supply a batch of examples in form of feed dict. If `is_val` flag is
        true, it will return validation data, otherwise, it returns training
        data.

        A feed_dict takes the form of:
        feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
            ....
        }

        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        # Create the feed_dict for the placeholders filled with the next
        # `batch size ` examples.
        batch_size = self.val_batch_size if self.is_val else self.batch_size
        images_feed, labels_feed = self.source.get_batch(batch_size,
                                                         self.is_val)
        feed_dict = {
            self.data: images_feed,
            self.labels: labels_feed,
        }
        return feed_dict


class TorchSensor(OldSensor):
    def __init__(self, pin_memory=False, **kwargs):
        super(TorchSensor, self).__init__(**kwargs)
        self.pin_memory = pin_memory

    def _setup(self):
        self.source.setup()
        # Legacy_TODO: try use pin memory.
        self.loader = th.utils.data.DataLoader(self.source.dataset,
                                               batch_size=self.batch_size,
                                               shuffle=self.shuffle_train,
                                               pin_memory=self.pin_memory,
                                               num_workers=8)
        self.val_loader = th.utils.data.DataLoader(self.source.val_dataset,
                                                   batch_size=self.val_batch_size,
                                                   pin_memory=self.pin_memory,
                                                   num_workers=8)
        self.iter = self.loader.__iter__()
        self.val_iter = self.val_loader.__iter__()

    def next(self):
        if self.is_val:
            try:
                return self.val_iter.next()
            except StopIteration:
                self.val_iter = self.val_loader.__iter__()
                return self.val_iter.next()
        else:
            try:
                return self.iter.next()
            except StopIteration:
                self.iter = self.loader.__iter__()
                A.inc_epoch()
                return self.iter.next()

    def _forward_train(self):
        ret = [th.autograd.Variable(t.cuda() if A.use_cuda() else t) for t in self.next()]
        A.cache_tensor_auto_scope(ret[0], "data")
        A.cache_tensor_auto_scope(ret[1], "labels")
        return ret

    def _forward_val(self):
        ret = [th.autograd.Variable(t.cuda() if A.use_cuda() else t) for t in self.next()]
        A.cache_tensor_auto_scope(ret[0], "val_data")
        A.cache_tensor_auto_scope(ret[1], "val_labels")
        return ret


__all__ = [name for name, x in locals().items() if
           not inspect.ismodule(x) and not inspect.isabstract(x)]
