"""
This module holds classes to model various kinds of data source a
`Sensor` may take as an input.

Some `Source`s only define more abstract methods, while some override
constructor method, since it needs more parameters. To create a concrete
`Source`, you use multiple inheritance to compose the `Source` you needs. The
first super class should be the one with the constructor you want, or in other
word, the one with the most complete parameters.
"""
import abc
import sys
import inspect

import tensorflow as tf

from .blocks import Block


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')


class Source(Block):
    """
    An abstract class to model data source from the world.

    The forms in which certain data arrived varies as many ways as one could
    imagine. This class abstracts this complexity and provides a uniform
    interface for subsequent processing.

    According to how tensorflow supplies data, two ways existed to
    correspondingly, `FeedSource` and `TFSource`.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 url,
                 work_dir="data",
                 validation_rate=None,
                 **kwargs):
        """
        Args:
            url: str
                Uniform Resource Locator. It could point to a file or web
                address, where this source should fetch data from.
            work_dir: str
                Working directory of the `Source`. It may puts temporal files,
                such as downloaded dataset and so on. By default, use a folder
                named data under root directory.
        validation_rate: a percentage
            The proportion of training data to be used as validation set. If
            None, all training data will be used for training.
        """
        super(Source, self).__init__(**kwargs)
        self.url = url
        self.work_dir = work_dir

        if validation_rate:
            assert validation_rate >= 0 and validation_rate < 1,\
                "The percentage used for validation must be between 0-1."
            self.validation_rate = validation_rate
        else:
            self.validation_rate = 0

    def data(self):
        """
        This is the first place a multiple outputs mechanism is
        needed. However, a temporary solution is used now, so this method is
        actually not used yet.
        """
        assert False, "Program should not reach here."

    @property
    def training_datum(self):
        """
        An optional property to provide a datum of particular training data
        source. It is not made an abstract method given some sub-classes are
        only able to provide shape information, aka `FeedSource`.
        """
        raise NotImplementedError("The property `training_datum` is not"
                                  " implemented!")
        sys.exit()

    @property
    def val_datum(self):
        """
        An optional property to provide a validation datum of particular data
        source. It is not made abstract for the same reason with
        `training_data`.
        """
        raise NotImplementedError("The property `val_datum` is not"
                                  "implemented!")
        sys.exit()

    @property
    def shape(self):
        """
        An optional property to provide shape of particular training data
        source. It is not made an abstract method since if some `Source` class
        returns data tensor directly, it already contains shape information.
        """
        raise NotImplementedError("The property `shape` is not implemented!")
        sys.exit()


class SupervisedSource(Source):
    """
    An abstract class to model supervised data source.
    """
    @property
    def label_shape(self):
        """
        An optional property to provide shape of particular label of training
        data source. It is not made an abstract method for the same reason with
        `shape` of `Source`.
        """
        raise NotImplementedError("The property `label_shape` is not"
                                  " implemented!")
        sys.exit()

    @property
    def training_label(self):
        """
        An optional property to provide a training label of particular data
        source. It is not made an abstract method given some sub-classes are
        only able to provide shape information.

        If a sub-class decides to use this property, the setup of tensor
        `labels` should be in `_setup`.
        """
        raise NotImplementedError("The property `training_label` is not"
                                  " implemented!")
        sys.exit()

    @property
    def val_label(self):
        """
        An optional property to provide a validation label of particular data
        source. It is not made abstract for the reason same with property
        `training_labels`.

        If a sub-class decides to use this property, the setup of tensor
        `labels` should be in `_setup`.
        """
        raise NotImplementedError("The property `val_label` is not"
                                  " implemented!")
        sys.exit()


class StaticSource(Source):
    """
    An abstract class to model static dataset partitioned into training data
    and test data.
    """
    def __init__(self, num_train, num_val, **kwargs):
        super(StaticSource, self).__init__(**kwargs)
        self.num_train = num_train
        self.num_val = num_val

    @property
    # TODO(Shuai): This property should be used to deal with the case batch
    # cannot divide the number of test data. It should be updated
    # accordingly. For instance, for InMemoryFeedSouce, the information comes
    # from `_epochs_completed` of `DataSet`.
    def epochs_completed(self):
        return self._epochs_completed


class FeedSource(Source):
    """
    An abstract class that supplies data in form of numpy.array.

    It does not create any tensor, and only plays the role to supply meta
    information of data to further `FeedSensor`, and of course supply actual
    data.

    Every concrete sub-class should implement a `get_batch` method to actually
    supply data, in the shape desired.
    """
    def __init__(self, center=False, scale=False, **kwargs):
        super(FeedSource, self).__init__(**kwargs)
        self.center = center
        self.scale = scale

    @abc.abstractmethod
    def get_batch(self, num, get_val):
        """
        Return `num` of datum, either in one numpy.array, or a tuple of
        numpy.array.

        Args:
            get_val: Boolean
                If True, get from validation samples or training samples.
        """
        raise NotImplementedError("Each sub `FeedSource` needs to implement"
                                  " this method to actually supply data.")
        sys.exit()


class InMemoryFeedSouce(StaticSource, FeedSource):
    """
    An abstract class to load all data into memory.

    This class holds a private member `_data_sets` to hold all data.
    """
    def _setup(self):
        """
        Call `_load` to load datasets into memory.
        """
        # Read the whole dateset into memory.
        self.data_sets = self._load()

    def get_batch(self, num, get_val):
        if get_val:
            return self.data_sets.test.next_batch(num)
        else:
            return self.data_sets.training.next_batch(num)

    def get_all(self, train):
        """
        Get all samples in the source.

        Args:
            train: Boolean
                Get from training samples or test samples.

        Returns:
            dataset: datasets.Dataset
        """
        if train:
            return self.data_sets.training
        else:
            return self.data_sets.test

    @abc.abstractmethod
    def _load(self):
        """
        Load data into memory.

        Returns:
            datasets: dataset.DataSets
        """
        raise NotImplementedError("Each sub `InMemorySouce` needs to implement"
                                  " this method to load data.")
        sys.exit()


class TFSource(StaticSource):
    """
    An abstract class that uses Reader Op of tensorflow to supply data.

    `_setup` of `TFSource` should initialized `self.data` to a tf.Tensor that
    returns by some Reader Op of tensorflow. So the data provided by this class
    of source has necessary information associated with it, and could be used
    directly in the further pipeline.

    Note the optional properties of `Source`, is made abstract, consequently
    mandatory.
    """
    def _setup(self):
        self._read()

    @abc.abstractmethod
    def _read(self):
        """
        TFSource uses Reader Ops of Tensorflow to read data. So any sub-classes
        of `TFSource` should implement it to actually read data. If it is
        combined with `SupervisedSource`, then setup of `labels` should also be
        put here.
        """
        raise NotImplementedError("Each sub-class of TFSource needs to"
                                  " implement this method to read data!")
        sys.exit()

    @abc.abstractproperty
    def training_datum(self):
        """
        An abstract property to enforce any sub-classes to provide training
        data in form of tf.Tensor.
        """
        raise NotImplementedError("Each sub-class of TFSource needs to"
                                  " implement this method to provide a"
                                  " training datum!")
        sys.exit()

    @abc.abstractproperty
    def val_datum(self):
        """
        An abstract property to enforce any sub-classes to provide training
        data in form of tf.Tensor.
        """
        raise NotImplementedError("Each sub-class of TFSource needs to"
                                  " implement this method to provide"
                                  " a validation datum!")
        sys.exit()


__all__ = [name for name, x in locals().items() if
           not inspect.ismodule(x) and not inspect.isabstract(x)]
