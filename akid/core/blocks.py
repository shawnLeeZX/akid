"""
`akid` builds another layer of abstraction on top of *Tensor*: *Block*.
Tensor can be taken as the media/formalism signal propagates in digital world,
while Block is the data processing entity that processes inputs and emits
outputs.

It coincides with a branch of "philosophy" called dataism that takes everything
in this world is a data processing entity. An interesting one that may come
from *A Brief History of Tomorrow* by Yuval Noah Harari.

Best designs mimic nature. `akid` tries to reproduce how signals in nature
propagates. Information flow can be abstracted as data propagating through
inter-connected blocks, each of which processes inputs and emits outputs. For
example, a vision classification system is a block that takes image inputs and
gives classification results. Everything is a `Block` in `akid`.

A block could be as simple as a convonlutional neural network layer that merely
does convolution on the input data and outputs the results; it also be as
complex as an acyclic graph that inter-connects blocks to build a neural
network, or sequentially linked block system that does data augmentation.

Compared with pure symbol computation approach, like the one in tensorflow, a
block is able to contain states associated with this processing unit. Signals
are passed between blocks in form of tensors or list of tensors. Many heavy
lifting has been done in the block (`Block` and its sub-classes),
e.g. pre-condition setup, name scope maintenance, copy functionality for
validation and copy functionality for distributed replicas, setting up and
gathering visualization summaries, centralization of variable allocation,
attaching debugging ops now and then etc.
"""
from __future__ import absolute_import, division, print_function

import abc
import copy

import tensorflow as tf

from ..utils import glog as log
from . import common
from .common import (
    TRAIN_SUMMARY_COLLECTION,
    VALID_SUMMARY_COLLECTION,
    SPARSITY_SUMMARY_SUFFIX
)


class Block(object):
    """
    The top level class. Everything should be its sub-class.
    """
    __metaclass__ = abc.ABCMeta

    def log(self, message, debug=False):
        """
        An unified place where logging should be processed. It is planned to
        handle logging level according to situation.

        Currently, it only handles info and debug level message, given the
        purpose of logging is to provide a detailed trace of operations of
        `akid`.

        NOTE: Remember to initialize it before using. See `utils.glog` for
        details.
        """
        if debug:
            log.debug(message)
        else:
            log.info(message)

    def error(self, message):
        log.error(message)

    def warning(self, message):
        log.warning(message)

    def get_copy(self):
        return copy.copy(self)


class ProcessingBlock(Block):
    """
    Abstract class for an arbitrary block that generates output. `Source`,
    `Sensor`, `ProcessingLayer`, `LossLayer` and `ProcessingSystem` etc are all
    sub-classes of this class.

    A `ProcessingBlock` should try to implement most of its functionality only
    with what it owns, and ask for communication (which in implementation is to
    provide interfaces) as little as possible.

    Outputs of a block is in form of properties. `ProcessingBlock` has an
    abstract property data which sub-class should implement to provide the
    processed outputs.

    `forward` is the interface for any containers, such as a `Brain` class,
    that hold this block, to call to do posterior probability inference (the
    statement may not hold exactly when the neural network does not learn a
    probability, but I believe in long term a unified view about what neural
    network is doing will agree with the ambiguity here). It is a wrapper for
    the actual abstract `_forward` method which should be implemented by
    concrete layer, and other pre-forward and post-forward methods. The caller
    is responsible for passing in the right data for the `forward` method.

    Call `forward` of each block before using it.
    """

    def __init__(self, do_summary=True, name=None, bag=None, **kwargs):
        """
        Create a layer and name it.

        Args:
            name: str
                The name of the layer. It also serves as the name scope of this
                layer in tensorflow. If `None`, the default name provided by
                Tensorflow will be used.
            do_summary: Boolean
                Whether to do summary on blocks contained. If True, outputs of
                this `ProcessingBlock` would be added to tensorflow summary.
            bag: supposed to be a dict
                A dictionary that holds any further ad hoc information one
                wants to keep in this block, such as a list of filters you want
                to visualize later.
        """
        super(ProcessingBlock, self).__init__(**kwargs)

        if not name:
            raise Exception(
                "{}'s `name` argument cannot be None! It serves as an"
                " identifier, also is used in visualization and"
                " summary etc.".format(type(self)))

        self.name = name
        self.do_summary = do_summary
        self.log("{} has bag: {}".format(name, bag))
        self.bag = bag

        # Variable scope to give unique names.
        self.var_scope = None

        # Boolean flag to indicate whether this layer has been built before. It
        # is used for determining whether we should share variables of this
        # layer later in `setup`. We need this for creating validation brain or
        # others network structures need sharing learnable variables.  It is
        # also useful to other things, such as determining whether to add
        # tensorboard summary etc.
        self.is_setup = None

    @abc.abstractmethod
    def data(self):
        """
        An abstract method to enforce all sub-classes to provide their
        processed data through this interface.

        Note that the data is the data obtained through `forward`.
        """
        raise NotImplementedError("Each concrete block needs to implement this"
                                  " method to provide an interface to offer"
                                  " data!")

    def forward(self, *args, **kwargs):
        """
        Common wrapper of all kinds of layers' `_forward` to do stat and
        visualization related logistics. If `_forward` has any output, it would
        be returned.

        A `ProcessingBlock` could be set up any times one wants. Each time it
        would build computational graph to process input provided this time,
        and any variables are shared.

        Args:
            All arguments will be passed to the actual `_forward` function.
        """
        if self.var_scope:
            var_scope = self.var_scope
        else:
            var_scope = self.name

        # Note an assignment operation will be wasted if self.var_scope already
        # has value, however, to make sure the first time assignment works,
        # this is the best way I can think of.
        with tf.variable_scope(var_scope) as self.var_scope:
            # TODO: the way how pre and post set up should be changed. Deferred
            # due to time budget. I am thinking actually implement it using
            # hooks, which makes things clearer. Or some part of the pre_setup
            # should be in a separate step, since setup now is changed to
            # forward.
            if not self._skip_pre_post_setup():
                self._pre_setup(*args, **kwargs)
            if not self._skip_pre_post_shared_setup():
                self._pre_setup_shared()
            self._forward(*args, **kwargs)
            if not self._skip_pre_post_setup():
                self._post_setup()
            if not self._skip_pre_post_shared_setup():
                self._post_setup_shared()

        self.is_setup = True
        self.var_scope.reuse_variables()

        return self.data

    def _pre_setup(self, *args, **kwargs):
        """
        Some setting won't be determined till the time to call setup. This is
        the place to set up the those settings. See `moving_averages` of
        `ProcessingLayer` for an example.
        """
        pass

    def _pre_setup_shared(self):
        """
        This is the place to do pre-setups on shared components. This method
        would only be called at the first time this block is setup. Refer to
        `_pre_setup` to see what pre-setups is for.
        """
        pass

    def _post_setup_shared(self):
        """
        This is the place to do post-setups on shared components. This method
        would only be called at the first time this block is setup. Refer to
        `_post_setup` to see what post-setups is for.
        """
        pass

    def _post_setup(self):
        """
        Some setting cannot be set up until the whole setup has been done. This
        is the place to set up those settings. See `moving_averages` of
        `ProcessingLayer` for an example.
        """
        pass

    def _skip_pre_post_setup(self):
        """
        Whether to skip `_pre_setup` and `_post_setup`. This method serves to
        provide a finer granularity control in `setup`. To change the behavior
        of any sub-classes, just override this method.
        """
        return False

    def _skip_pre_post_shared_setup(self):
        """
        Whether to skip `_pre_setup_shared` and `_post_setup_shared`. This
        method serves to provide a finer granularity control in `setup`. To
        change the behavior of any sub-classes, just override this method.
        """
        return self.is_setup

    @abc.abstractmethod
    def _forward(self):
        """
        An abstract method that must be overrided.
        """
        raise NotImplementedError('Each sub-layer needs to implement this'
                                  'method to process data!')


class ShadowableBlock(ProcessingBlock):
    """
    A block for creating shadow replicas for parallelism.

    A shadow replica is a replica of the same parameter of the genuine one, but
    be placed on different device for parallelism. An example would be data
    parallelism in multiple GPU setting or distributed setting.
    """
    def __init__(self, **kwargs):
        super(ShadowableBlock, self).__init__(**kwargs)

        # Whether this processing block is a shallow replica.
        self.is_shadow = False

    def _skip_pre_post_setup(self):
        return self.is_shadow

    def set_shadow(self):
        """
        Set this block to a shadow replica.
        """
        self.is_shadow = True

    def get_shadow_copy(self):
        shadow_copy = self.get_copy()
        shadow_copy.set_shadow()
        return shadow_copy

    def log(self, msg, *args, **kwargs):
        """
        Logging method to control the verbosity of the output. It logs all
        logging of shadow replica to debug.
        """
        # It is possible the constructor of `ShadowableBlock` has not been
        # called, in this case we fall back to its super's log.
        if hasattr(self, "is_shadow") and self.is_shadow:
            log.debug(msg)
        else:
            super(ShadowableBlock, self).log(msg, *args, **kwargs)


class ProcessingLayer(ShadowableBlock):
    """
    An abstract layer for data processing layer in the brain.

    A `ProcessingLayer` is nothing if it does not process data. So every
    sub-class of `ProcessingLayer` should possess a `data` property as
    interface to provide the data it processes.

    For now, all processing layers only have one output, and provide it via
    property `data`. So it overrides the `data` method of `ProcessingBlock`.

    Optionally a `ProcessingLayer` could have a `loss` property for loss (or
    losses) in this layer and a `eval` property for any evaluation metrics in
    this layer. `loss` should be a list if there are multiple losses, so is
    eval graphs.

    If `inputs` in the constructor is not None (in this case it should be a
    list), this layer is supposed to have multiple inputs. Refer to
    `system.GraphSystem` for more explanation.
    """
    def __init__(self, moving_average_decay=None, inputs=None, **kwargs):
        """
        Args:
            moving_average_decay: A fraction. If `None`, When the parameters of
                this layer is being shared, the shared paras should be
                the current value of the paras. If has a value, it would be
                used in `tf.train.ExponentialMovingAverage`. Then the shared
                value would be the moving average.
            inputs: list
                A list to list inputs of this layer. Refer to
                `system.GraphSystem` for more explanation.
        """
        super(ProcessingLayer, self).__init__(**kwargs)

        assert moving_average_decay is None or \
            moving_average_decay >= 0.5 and moving_average_decay < 1, \
            ("Invalid moving_average_decay value {}. Should be None or"
             " between [0.5, 1]".format(moving_average_decay))
        self.moving_average_decay = moving_average_decay

        self.inputs = inputs

        # Bookkeeping all variables.
        self.var_list = []

        # A Boolean flag to indicate whether this block is in validation mode.
        self.is_val = False

    def get_val_copy(self):
        """
        Get a copy for validation.

        Since a processing layer is learned, it has to be taken out for
        evaluation from time to time.
        """
        val_copy = self.get_copy()
        val_copy.set_val()
        return val_copy

    def set_val(self):
        self.is_val = True

    def _pre_setup(self, *arg, **kwargs):
        super(ProcessingLayer, self)._pre_setup()
        if self.is_val:
            self.var_scope.reuse_variables()

    def _pre_setup_shared(self):
        # Moving averages are supposed be shared so it would only be set up
        # once.
        if self.moving_average_decay:
            # We pass current training step to moving average to speed up
            # updates moving average of variables at the beginning of the
            # training since moving average is useful only later.
            with tf.variable_scope(common.global_var_scope, reuse=True):
                step = tf.get_variable(common.GLOBAL_STEP)
            self.moving_averages = tf.train.ExponentialMovingAverage(
                self.moving_average_decay, step)

    def _post_setup(self):
        # TODO: ideally, we want to control how many summaries we gather.
        if self.do_summary:
            log.info("Do tensorboard summary on outputs of {}".format(
                self.name))
            collection_to_add = VALID_SUMMARY_COLLECTION if self.is_val \
                else TRAIN_SUMMARY_COLLECTION
            if self.data is not None:
                if type(self.data) is not list:
                    self._data_summary(self.data, collection_to_add)
            if self.loss is not None:
                tf.summary.scalar(self.loss.op.name,
                                  self.loss,
                                  collections=[collection_to_add])
            if self.eval is not None:
                if type(self.eval) is list:
                    for e in self.eval:
                        tf.summary.scalar(e.op.name,
                                          e,
                                          collections=[collection_to_add])
                else:
                    tf.summary.scalar(self.eval.op.name,
                                      self.eval,
                                      collections=[collection_to_add])

    def _data_summary(self, data, collection=TRAIN_SUMMARY_COLLECTION):
        """
        Helper function to do statistical summary on the bundle of data.

        Args:
            collection: which op collection to add to. It should be one of
                TRAIN_SUMMARY_COLLECTION or VALID_SUMMARY_COLLECTION from
                akid.core.common.
        """
        assert collection is TRAIN_SUMMARY_COLLECTION or \
            collection is VALID_SUMMARY_COLLECTION, \
            "{} is not one of those defined in common.py. Some thing is wrong"
        tf.summary.histogram(data.op.name + '/activations',
                             data,
                             collections=[collection])
        tf.summary.scalar(data.op.name + '/' + SPARSITY_SUMMARY_SUFFIX,
                          tf.nn.zero_fraction(data),
                          collections=[collection])

    def _post_setup_shared(self):
        # Maintain moving averages of variables.
        if self.moving_average_decay and len(self.var_list) is not 0:
            self.moving_averages_op = self.moving_averages.apply(self.var_list)
            with tf.control_dependencies([self.moving_averages_op]):
                self._data = tf.identity(
                    self._data,
                    # We add one underscore to the original data's name to the
                    # has-to existing identity data due to the need of control
                    # dependency.
                    name=self._data.op.name.split('/')[-1] + "_")

        if self.do_summary:
            log.info("Do tensorboard summary on variables of {}".format(
                self.name))
            for var in self.var_list:
                self._var_summary(var.op.name, var)
        if self.moving_average_decay:
            for var in self.var_list:
                var_average = self.moving_averages.average(var)
                self._var_summary(var.op.name + "_average", var_average)

        # Log parameter number of this layer.
        total_para_num = 0
        for var in self.var_list:
            shape = var.get_shape().as_list()
            para_num = 1
            for dim in shape:
                para_num *= dim
            total_para_num += para_num
        log.info("This layer has {} parameters.".format(total_para_num))

    def _var_summary(self, tag, var):
        if len(var.get_shape().as_list()) is 0:
            tf.summary.scalar(tag, var, collections=[TRAIN_SUMMARY_COLLECTION])
        else:
            tf.summary.histogram(tag,
                                 var,
                                 collections=[TRAIN_SUMMARY_COLLECTION])

    def _skip_pre_post_shared_setup(self):
        """
        Whether to skip `_pre_setup_shared` and `_post_setup_shared`. This
        method serves to provide a finer granularity control in `setup`. To
        change the behavior of any sub-classes, just override this method.
        """
        return self.is_setup or self.is_val

    @property
    def data(self):
        """
        All sub-class `ProcessingLayer` should save the processed data to
        `_data`.
        """
        if hasattr(self, "_data"):
            return self._data
        else:
            return None

    @property
    def loss(self):
        """
        Each `ProcessingLayer` could optionally associated it with a loss. A
        `ProcessingLayer` is the smallest unit in the hierarchical data
        processing system. Since for every data processing system, it would has
        a loss, which is the purpose of its existence, so a `ProcessingLayer`
        layer also could have a loss.

        All sub-classes should save loss to `_loss`.
        """
        if hasattr(self, "_loss"):
            return self._loss
        else:
            return None

    @property
    def eval(self):
        """
        All sub-class `ProcessingLayer` should save the evaluation graph to
        `_eval`.
        """
        if hasattr(self, "_eval"):
            return self._eval
        else:
            return None

    @property
    def train_op(self):
        """
        Ops that need called along with the top level train op. Sub-class
        should save the train op to `_train_op`.
        """
        if hasattr(self, "_train_op"):
            return self._train_op
        else:
            return None

    def _get_variable(self, name, shape, initializer, trainable=True):
        """
        Allocate or retrieve tensorflow variables. If the variable has already
        existed, depending on `moving_average_decay`'s value, moving average of
        it(when `moving_average_decay` has a value) or the original
        variable(when `moving_average_decay` is None) would be returned. Refer
        to `tf.get_variable()` to the details of a shared variable in
        tensorflow.
        """
        var = tf.get_variable(name,
                              shape,
                              initializer=initializer,
                              trainable=trainable)

        if self.is_setup:
            if self.moving_average_decay:
                log.debug("Use moving average of paras {}".format(
                    var.op.name))
                var = self.moving_averages.average(var)
            else:
                log.debug("Reuse paras {}".format(var.op.name))
        else:
            # Append it to the var list, do moving average later in
            # `_post_setup`.
            self.var_list.append(var)

        return var
