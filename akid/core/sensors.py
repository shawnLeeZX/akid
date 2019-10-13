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

Sensors can also include the support for functions such as multi-threaded, or
multi-process data prefetch to build better performed data preparation
pipeline.

Mathematically, it is a system made up with a series of linked blocks that do
data augmentation.

As an example, again saying in supervised setting, a sensor is a block that
takes a data source and output sensed (batched and augmented) data. A sensor
needs to be used along with a source. A concrete example could be the sensor
for the MNIST dataset. Taking a `Source`, we could make a sensor::

      sensor = SimpleSensor(name='data', source_in=source)

Similar with `Source`, a sensor has multiple modes to supply data. Refer to the
documentation of `Source` for more details.

------------------
Sensor as iterator
------------------

A `Sensor` can be used as an iterator when an automatic precise control on many
epochs of data should be provided is needed. To use sensor as an iterator::

    for batch in sensor:
        # blah blah

Note that, even if a `Sensor` is used as an iterator, one should call `setup`
and `teardown` before a sensor is used and after it finishes its role. This is
to properly shutdown threads, or processes that are prefetching data, if there
are any.

NOTE: Some sensors may do not support iteration. Check documentation for details.
"""
from __future__ import absolute_import, division, print_function

import sys
import abc
import os
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
from torch import multiprocessing as mp

from .jokers import JokerSystem
from .blocks import ValidatableProcessingBlock
from . import sources
from . import samplers
from .common import TRAIN_SUMMARY_COLLECTION, VALID_SUMMARY_COLLECTION
from .events import (
    DataPrefetchThreadsDeadEvent,
    DataPrefetchProcessesDeadEvent,
    EpochCompletedEvent
)
from akid import backend as A
from akid.utils import glog as log


def _check_if_main_thread_is_dead():
    # If the main thread is dead, quit.
    for t in threading.enumerate():
        if t.name == "MainThread":
            if not t.is_alive():
                return True
            else:
                return False


def _data_fetching_worker(source, index_queue, data_queue, done_event):
    # If index queue is not empty, then fetch the indices of a batch, and load
    # the data to data queue.
    while True:
        if _check_if_main_thread_is_dead():
            return

        # If the done event is set, quit.
        if done_event.is_set():
            return

        try:
            log.debug("Data Prefetching Worker: Fetching data according to indices ... ")
            indices = index_queue.get(timeout=A.TIMEOUT)
            data = source.get(indices)
            data_gpu = [A.Tensor(d) for d in data]
            data_queue.put(data_gpu, timeout=A.TIMEOUT)
            log.debug("Data Prefetching Worker: Data fetched")
        except queue.Empty:
            continue


def _data_fetching_worker_process(i, source, index_queue, data_queue, done_event):
    repeat = False
    NAME = "Data Prefetching Worker {}".format(i)

    while True:
        log.debug("{}: Working ...".format(NAME))
        try:
            os.kill(os.getppid(), 0)
        except ProcessLookupError:
            return

        # If the done event is set, quit.
        log.debug("{}: Done event {}".format(NAME, done_event.is_set()))
        if done_event.is_set():
            return

        try:
            log.debug("{}: Fetching data according to indices ... ".format(NAME))
            if not repeat:
                # It is possible the previous try to enqueue data is timeout,
                # so we do not fetch data this time.
                item = index_queue.get(timeout=A.TIMEOUT)
                if type(item) is EpochCompletedEvent:
                    # Epoch finishes, we pass on the event to let the sensor
                    # knows an epoch has indeed finished.
                    data = item
                else:
                    data = source.get(item) # The item is a list of indices now.
            data_queue.put(data, timeout=A.TIMEOUT)
            repeat = False
            log.debug("{}: Data fetched".format(NAME))
        except queue.Empty:
            continue
        except queue.Full:
            repeat = True
            continue
        except Exception as e:
            log.debug("Process {} : Exception {}.".format(i, e))
            raise e


def _data_preloading_worker(prefetch_data_queue, data_queue, done_event):
    log.debug("Data Preloading Worker: Started. ")
    NAME = "Data Preloading Worker"

    repeat = False

    while True:
        log.debug("Data Preloading Worker: Working ... ")
        if _check_if_main_thread_is_dead():
            return

        try:
            log.debug("{}: Loading CPU data ... ".format(NAME))

            if not repeat or done_event.is_set():
                data = prefetch_data_queue.get(timeout=A.TIMEOUT)
                if data is None:
                    assert done_event.is_set()
                    return
                elif done_event.is_set():
                    # We need to consume all data before finishing, otherwise,
                    # data fetching workers may cannot finish since it waits to
                    # enqueue data (for that the pipe used by the queue may be
                    # full, thus the workers are waiting the data to be
                    # consumed; this is an implementation issue of
                    # multiprocessing.Queue in python). But we do not need to
                    # process them anymore.
                    continue
                elif type(data) is list:
                    data_gpu = []
                    for d in data:
                        data_gpu.append(A.Tensor(d))

            log.debug("{}: Enqueuing GPU data ... ".format(NAME))

            if type(data) is EpochCompletedEvent:
                data_queue.put(data, timeout=A.TIMEOUT)
            else:
                data_queue.put(data_gpu, timeout=A.TIMEOUT)
            repeat = False

            log.debug("Data Preloading Worker: Data loaded")
        except queue.Empty:
            log.debug("{}: Queue is empty. Repeat.".format(NAME))
            continue
        except queue.Full:
            repeat = True
            continue


class Sensor(ValidatableProcessingBlock):
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
                 val_sampler="sequence",
                 **kwargs):
        """
        Args:
            source_in: Source
                Where data should be sensed from.
            batch_size: int
                The number of samples in a time the sensor would provide.
            queue_size: int
                The number of batches to prefetch.
            sampler: str or a Sampler object
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
        if self.queue_size > self.num_batches_per_epoch:
            self.queue_size = self.num_batches_per_epoch

        # More complex datasets might customized samplers, Sensor supports pass
        # a built sampler directly.
        if type(sampler) is str:
            self.train_sampler_name = sampler
        else:
            self.train_sampler = sampler

        if type(val_sampler) is str:
            self.val_sampler_name = val_sampler
        else:
            self.val_sampler = val_sampler

        self.mode = A.Mode.TRAIN

    @property
    def batch_size(self):
        return self.batch_size_dict[self.mode]

    @property
    def num_batches_per_epoch(self):
        return (self.source.size - 1) // self.batch_size_dict[self.mode] + 1

    def set_batch_size(self, size):
        """
        NOTE: the queue size is determined dynamically from batch
        size. However, if batch size is set after sensor setup, the queue size
        is determined actually using the old batch size value.
        """
        if type(size) is not int:
            raise ValueError("The batch size should be of type int. Type {} received".format(type(size)))
        else:
            self.batch_size_dict[self.mode] = size
            if self.queue_size > self.num_batches_per_epoch:
                self.queue_size = self.num_batches_per_epoch

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
        self.log("Mode {}".format(mode))

    def reset(self):
        if self.is_setup:
            self._teardown_data_queue()
            self.sampler.reset()

        self.setup()

    def _setup(self):
        """
        Set up a FIFO queue to load data from source.
        """
        self.source.setup()
        if self.mode == A.Mode.TRAIN:
            if hasattr(self, "train_sampler"):
                self.sampler = self.train_sampler
            else:
                self.sampler = samplers.get(self.train_sampler_name, self.source.size)
        elif self.mode == A.Mode.VAL or self.mode == A.Mode.TEST:
            if hasattr(self, "val_sampler"):
                self.sampler = self.val_sampler
            else:
                self.sampler = samplers.get(self.val_sampler_name, self.source.size)
        else:
            raise ValueError("Wrong mode {}".format(self.mode))

        self._setup_index_queue()
        self._setup_data_queue()

    def teardown(self):
        self._teardown_data_queue()

    @abc.abstractmethod
    def _setup_index_queue(self):
        """
        The procedure to set up an index queue.
        """
        pass

    @abc.abstractproperty
    def index_queue(self):
        """
        Any subclass should has a queue to provide indices for workers to
        fetch.
        """
        pass

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
        Any subclass should has a data queue to hold data fetched according to
        index queue.
        """
        pass

    def _forward(self, *args, **kwargs):
        if self.index_queue.empty():
            try:
                # If we just finishes using the sensor as an iterator, calling
                # forward now would results in errors, since no data is in the data
                # queue now. If this is the case, we need to put same indices to
                # prefetch to continue.
                self.index_queue.put(self.sampler.next(self.batch_size_dict[self.mode]))
            except EpochCompletedEvent:
                # In some cases, it is possible that an epoch is finished, yet
                # the next epoch has not been started. In such cases, we just
                # keep fetching.
                self.index_queue.put(self.sampler.next(self.batch_size_dict[self.mode]))

        ret = self.data_queue.get()

        A.cache_tensor_auto_scope(ret[0], "val_data" if self.is_val else "data")
        A.cache_tensor_auto_scope(ret[1], "val_labels" if self.is_val else "labels")
        self._data = ret

        # It looks like the first index queue refill could be merged with the
        # second one below. But it would require the index queue size is not
        # fill before each data queue fetch. Though this is an easy condition
        # to meet, I would like to just keep it in the current way.
        try:
            self.index_queue.put(self.sampler.next(self.batch_size_dict[self.mode]))
        except EpochCompletedEvent:
            # Just keep prefetching the next batch.
            self.index_queue.put(self.sampler.next(self.batch_size_dict[self.mode]))

        return self._data

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


class SimpleSensor(Sensor):
    """
    A simple sensor that uses a single thread to prefetch data.
    """
    def _pre_forward(self):
        super(SimpleSensor, self)._pre_forward()
        if threading.active_count() == 1:
            # Only the main thread is alive. Raise exception.
            raise DataPrefetchThreadsDeadEvent

    def _setup_index_queue(self):
        # Set up a queue.
        self._index_queue = Queue(self.queue_size)
        # Start loading data from source according to mode.
        ## Put indices to prefetch.
        for i in range(self.queue_size):
            try:
                self._index_queue.put(self.sampler.next(self.batch_size_dict[self.mode]))
            except EpochCompletedEvent:
                self._index_queue.put(self.sampler.next(self.batch_size_dict[self.mode]))

    @property
    def index_queue(self):
        return self._index_queue

    def _setup_data_queue(self):
        # Set up a data queue.
        self._data_queue = Queue(self.queue_size)
        # Start loading data from source according to mode.
        ## Start the workers to fetch data.
        self.done_event = threading.Event()
        self.worker_thread = threading.Thread(
            target=_data_fetching_worker,
            args=(self.source, self._index_queue, self._data_queue, self.done_event))
        self.worker_thread.start()

    def _teardown_data_queue(self):
        self.done_event.set()
        self.worker_thread.join()

    @property
    def data_queue(self):
        return self._data_queue


class ParallelSensor(Sensor):
    """
    The sensor that fetches data using multiple processes. It supports to be
    used as an iterator.
    """
    def __init__(self, num_workers=4, *args, **kwargs):
        super(ParallelSensor, self).__init__(*args, **kwargs)
        self.num_workers = num_workers
        self.queue_size *= num_workers
        if self.queue_size > self.num_batches_per_epoch:
            self.queue_size = self.num_batches_per_epoch

    def __iter__(self):
        self.epoch_finished = False
        if self.index_queue.empty():
            try:
                # Upon setup of sensor, we have some indices put in the queue, so
                # it may not be necessary put some indices. However, after the
                # second entry to create an iterator, it is possible the index
                # queue is empty, since further prefetching is disabled. In such a
                # case, we need to put some indices to prefetch to get started.
                self.index_queue.put(self.sampler.next(self.batch_size_dict[self.mode]))
            except EpochCompletedEvent as e:
                # If the dataset is small, we could finish the dataset while we
                # are creating the iterator (since we are prefetching data upon
                # setup). In such a case, we would call it a day.
                self.epoch_finished = True
                self.index_queue.put(e)

        return self

    def __next__(self):
        # Get the next batch of data.
        ret = self.data_queue.get(timeout=A.TIMEOUT)

        # Raise `StopIteration` since we have received the epoch completion
        # signal we sent earlier.
        if type(ret) is EpochCompletedEvent:
            raise StopIteration

        # If not, we do our normal data stuff.
        A.cache_tensor_auto_scope(ret[0], "val_data" if self.is_val else "data")
        A.cache_tensor_auto_scope(ret[1], "val_labels" if self.is_val else "labels")
        self._data = ret

        # Put the indices of the batch to be prefetched if we still have not
        # finished an epoch. But if we have, stop fetching data by putting the
        # EpochCompletedEvent to the index queue.
        if not self.epoch_finished:
            try:
                self.index_queue.put(self.sampler.next(self.batch_size_dict[self.mode]))
            except EpochCompletedEvent as e:
                self.epoch_finished = True
                self.index_queue.put(e)

        return ret

    def _pre_forward(self):
        super(ParallelSensor, self)._pre_forward()
        if len(mp.active_children()) == 0:
            # If all child processes are dead, we raise error.
            raise DataPrefetchProcessesDeadEvent
        if not self.preloading_thread.is_alive():
            # The thread to move data from CPU to GPU is dead.
            raise DataPrefetchThreadsDeadEvent

    def _setup_index_queue(self):
        self._index_queue = mp.Queue(self.queue_size)
        for i in range(self.queue_size):
            try:
                self._index_queue.put(self.sampler.next(self.batch_size_dict[self.mode]))
            except EpochCompletedEvent:
                self._index_queue.put(self.sampler.next(self.batch_size_dict[self.mode]))

    @property
    def index_queue(self):
        return self._index_queue

    def _setup_data_queue(self):
        # Set up a data queue.
        self._prefetch_data_queue = mp.Queue(self.queue_size)
        # Start loading data from source according to mode.
        ## Start the workers to fetch data.
        self.done_event = mp.Event()
        self.worker_processes = []
        for i in range(self.num_workers):
            process = mp.Process(
                target=_data_fetching_worker_process,
                args=(i, self.source, self._index_queue, self._prefetch_data_queue, self.done_event))
            process.start()
            self.worker_processes.append(process)

        # Start a thread to load the CPU data to GPU.
        self._data_queue = Queue(self.queue_size * 2)
        self.preloading_thread = threading.Thread(
            target=_data_preloading_worker,
            args=(self._prefetch_data_queue, self._data_queue, self.done_event))
        self.preloading_thread.start()

    def _teardown_data_queue(self):
        self.done_event.set()
        for i in range(self.num_workers):
            self.log("Waiting worker {} to join ...".format(i))
            self.worker_processes[i].join()
        # When training on MNIST, the preloading_thread dies each time data
        # queues are torn down. To prevent deadlock, check whether it is dead
        # or not. Why it dies is not clear.
        if self.preloading_thread.is_alive():
            self._prefetch_data_queue.put(None)
            self.log("Waiting preloading thread to join ...")
            self.preloading_thread.join()
        else:
            self.log("Preloading thread is dead already.")

    @property
    def data_queue(self):
        return self._data_queue


@deprecated(reason="Legacy code. Use `Sensor` instead.")
class OldSensor(six.with_metaclass(abc.ABCMeta, ValidatableProcessingBlock)):
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
