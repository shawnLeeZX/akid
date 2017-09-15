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

Two types of blocks exist: the blocks build computational graph, and does not
care about execution; and the blocks that deal with execution (TODO). The later
part has not been sorted through yet.

The type one block could be as simple as a convonlutional neural network layer
that merely does convolution on the input data and outputs the results; it also
be as complex as an acyclic graph that inter-connects blocks to build a neural
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
    SPARSITY_SUMMARY_SUFFIX,
    FILTER_WEIGHT_COLLECTION,
)
from ..core import initializers
from .. import backend as A


class Block(object):
    """
    The top level class. Everything should be its sub-class.

    A `Block` holds computational graph. Sub-class should implement `_setup` to
    build the graph. It does not enforce how the computational graph should be
    used, which should be the responsibility of sub classes.

    `Block` supports add arbitrary code after setup by offering
    `post_setup_hook`. All functions added in the hook will be called after
    setup. `self` is passed as an argument to functions in the hook.

    Call `setup` of each block before using it.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, name=None):
        self.name = name

        if not name:
            raise Exception(
                "{}'s `name` argument cannot be None! It serves as an"
                " identifier, also is used in visualization and"
                " summary etc.".format(type(self)))

        # Hooks that are called after setup.
        self.post_setup_hook = []
        self.pre_setup_hook = []

        self.is_setup = None

    def setup(self):
        with A.variable_scope(self.name):
            self._pre_setup()
            self._setup()
            self._post_setup()

        self.is_setup = True

    def _setup(self):
        pass

    def _pre_setup(self):
        for f in self.pre_setup_hook:
            f(self)

    def _post_setup(self):
        for f in self.post_setup_hook:
            f(self)

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

    `ProcessingBlock` builds computational graph that processes data.

    A `ProcessingBlock` should try to implement most of its functionality only
    with what it owns, and ask for communication (which in implementation is to
    provide interfaces) as little as possible.

    Outputs of a block is in form of properties. `ProcessingBlock` has an
    abstract property data which sub-class should implement to provide the
    processed outputs.

    `forward` is the interface for any containers, such as a `Brain` class,
    that hold this block, to call to do build computational graph that normally
    leads to a forward propagation in neural networks to do posterior inference
    (the statement may not hold exactly when the neural network does not learn
    a probability). The concrete forward propagation of sub-classes should be
    implemented in `_forward` method. To add arbitrary code before and after
    `_forward`, use `pre_forward_hook`; to add after, use
    `post_forward_hook`. All arguments passed to `_forward` are also available
    in functions in the hooks.
    """

    def __init__(self, do_summary=True, bag=None, **kwargs):
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

        self.do_summary = do_summary
        self.log("{} has bag: {}".format(self.name, bag))
        self.bag = bag

        # Hooks
        self.pre_forward_hook = []
        self.post_forward_hook = []

        # Some operations only run at the first forward pass, so set a flag.
        self.done_first_pass = False

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

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Args:
            All arguments will be passed to the actual `_forward` function.
        """
        if not self.is_setup:
            self.setup()

        with A.variable_scope(self.name):
            self._pre_forward(*args, **kwargs)
            self._forward(*args, **kwargs)
            self._post_forward(*args, **kwargs)
            if not self.done_first_pass:
                self._first_forward_logistics(*args, **kwargs)

        return self.data

    def _pre_forward(self, *args, **kwargs):
        for f in self.pre_forward_hook:
            f(*args, **kwargs)

    def _post_forward(self, *args, **kwargs):
        for f in self.post_forward_hook:
            f(*args, **kwargs)

    def _first_forward_logistics(self, *args, **kwargs):
        """
        During the first pass, some operations, e.g. summary ops, may be
        created, but will not run in each forward. These operations should be
        create during the first pass, and be gathered somehow.
        """
        pass

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

    def _pre_forward(self, *args, **kwargs):
        if not self.is_shadow:
            for f in self.pre_forward_hook:
                f(*args, **kwargs)

    def _post_forward(self, *args, **kwargs):
        if not self.is_shadow:
            for f in self.post_forward_hook:
                f(*args, **kwargs)

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


class GenerativeBlock(ShadowableBlock):
    """
    Block that optionally supports a backward pass.

    It aims to model generative modeling instead of just discriminative
    training. It supports a `backward` that normally does generative
    reconstruction according to the forward propagation (posterior inference)
    results.
    """
    def backward(self, X):
        """
        According to the top-down inference results, reconstruct the input.

        TODO: should call _backward somehow maybe
        """
        raise NotImplementedError("This block does not support backward pass.")

    @property
    def data_g(self):
        if hasattr(self, "_data_g"):
            return self._data_g
        else:
            raise NotImplemented("This block does not have generated data.")


class ProcessingLayer(GenerativeBlock):
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
    def __init__(self,
                 moving_average_decay=None,
                 inputs=None,
                 wd={"type": "l2", "scale": 5e-4},
                 do_summary_on_val=False,
                 **kwargs):
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
            wd: dict
                An dictionary that contains the `type` of regularization to use
                and its scale, which is to say the multiplier of the
                regularization loss term. If None, weight decay is not added.
            do_summary_on_val: bool
                Whether to do summary on validation copy.
        """
        super(ProcessingLayer, self).__init__(**kwargs)

        assert moving_average_decay is None or \
            moving_average_decay >= 0.5 and moving_average_decay < 1, \
            ("Invalid moving_average_decay value {}. Should be None or"
             " between [0.5, 1]".format(moving_average_decay))
        self.moving_average_decay = moving_average_decay

        self.inputs = inputs
        self.wd = wd
        self.do_summary_on_val = do_summary_on_val

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

    def _variable_with_weight_decay(self, name, shape, init_para=None):
        """Helper to create an initialized Variable with weight decay.

        Args:
            name: name of the variable
            shape: list of ints

        Returns:
            (Variable Tensor, Weight Decay Loss) If `self.wd` is `None`, then
            the returned weight decay loss would be `None`.
        """
        if not init_para:
            init_para = self.init_para

        var = self._get_variable(name, shape, self._get_initializer(init_para))
        if len(shape) > 1:
            # Add non-bias filters to the collection.
            tf.add_to_collection(FILTER_WEIGHT_COLLECTION, var)

        weight_decay = None
        if self.wd and self.wd["scale"] is not 0:
            try:
                self.log("Using {} regularization with scale {}".format(
                    self.wd["type"], self.wd["scale"]))

                if self.wd["type"] == "l2":
                    weight_decay = A.mul(A.nn.l2_loss(var),
                                         self.wd["scale"],
                                         name=name + '/l2_loss')
                elif self.wd["type"] == "l1":
                    weight_decay = A.mul(A.nn.l1_loss(var),
                                         self.wd["scale"],
                                         name=name + '/l1_loss')
                else:
                    raise Exception("Type {} loss is not supported!".format(
                        self.wd["type"]))
            except KeyError as e:
                raise Exception("`{}` not found in the provided regularization"
                                " parameters, `wd`. Perhaps you have some"
                                " typos.".format(e.message))

        return var, weight_decay

    def _get_initializer(self, init_para=None):
        if not init_para:
            init = initializers.get("default")
        else:
            name = init_para["name"]
            kwargs = init_para.copy()
            kwargs.pop("name")
            init = initializers.get(name, **kwargs)

        self.log("Variables of {} uses initializer {} with arguments {}".format(
            self.name, name, kwargs))

        return init

    def _pre_setup(self):
        super(ProcessingLayer, self)._pre_setup()

        if self.is_val:
            self.var_scope.reuse_variables()

        if self.moving_average_decay:
            # We pass current training step to moving average to speed up
            # updates moving average of variables at the beginning of the
            # training since moving average is useful only later.
            step = A.get_step()
            self.moving_averages = tf.train.ExponentialMovingAverage(
                self.moving_average_decay, step)

    def _first_forward_logistics(self, *args, **kwargs):
        super(ProcessingLayer, self)._first_forward_logistics(*args, **kwargs)

        if not self.do_summary:
            return

        if not self.is_val or self.is_val and self.do_summary_on_val:
            self.log("Do tensorboard summary on outputs of {}".format(
                self.name))

            collection_to_add = VALID_SUMMARY_COLLECTION if self.is_val \
                else TRAIN_SUMMARY_COLLECTION
            if self.data is not None:
                if type(self.data) is not list:
                    name = A.get_name(self.data)
                    if name:
                        self._data_summary(self.data, collection_to_add)

            if self.loss is not None:
                name =  A.get_name(self.loss)
                if name:
                    A.summary.scalar(name,
                                     self.loss,
                                     collections=[collection_to_add])
            if self.eval is not None:
                if type(self.eval) is list:
                    for e in self.eval:
                        name = A.get_name(e)
                        if name:
                            A.summary.scalar(name,
                                            e,
                                             collections=[collection_to_add])
                else:
                    name = A.get_name(self.eval)
                    if name:
                        A.summary.scalar(name,
                                         self.eval,
                                         collections=[collection_to_add])

    def _data_summary(self, data, collection=TRAIN_SUMMARY_COLLECTION, sparsity_summary=True):
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
        if type(data) is not list and type(data) is not tuple:
            data = [data]

        for d in data:
            name = A.get_name(d)
            if name:
                A.summary.histogram(name + '/activations',
                                    d,
                                    collections=[collection])
                if sparsity_summary:
                    A.summary.scalar(name + '/' + SPARSITY_SUMMARY_SUFFIX,
                                     A.nn.zero_fraction(d),
                                     collections=[collection])

    def _post_setup(self):
        super(ProcessingLayer, self)._post_setup()
        # Maintain moving averages of variables.
        if self.moving_average_decay and len(self.var_list) is not 0:
            self.moving_averages_op = self.moving_averages.apply(self.var_list)

            for var in self.var_list:
                var_average = self.moving_averages.average(var)
                self._var_summary(A.get_name(var) + "_average", var_average)

        if self.do_summary:
            log.info("Do tensorboard summary on variables of {}".format(
                self.name))
            for var in self.var_list:
                self._var_summary(A.get_name(var), var)

        # Log parameter number of this layer.
        total_para_num = 0
        for var in self.var_list:
            shape = A.get_shape(var)
            para_num = 1
            for dim in shape:
                para_num *= dim
            total_para_num += para_num
        log.info("This layer has {} parameters.".format(total_para_num))

    def on_para_update(self):
        """
        Operations to run after parameter update.

        Returns:
            A list of ops.
        """
        if hasattr(self, "moving_average_op"):
            return [self.moving_averages_op]
        else:
            return []

    def _var_summary(self, tag, var):
        if len(A.get_shape(var)) is 0:
            A.summary.scalar(tag,
                             var,
                             collections=[TRAIN_SUMMARY_COLLECTION])
        else:
            A.summary.histogram(tag,
                                var,
                                collections=[TRAIN_SUMMARY_COLLECTION])

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
        var = A.get_variable(name,
                             shape,
                             initializer=initializer,
                             trainable=trainable)

        if self.is_setup:
            if self.moving_average_decay:
                log.debug("Use moving average of paras {}".format(
                    A.get_name(var)))
                var = self.moving_averages.average(var)
            else:
                log.debug("Reuse paras {}".format(A.get_name(var)))
        else:
            # Append it to the var list, do moving average later in
            # `_post_setup`.
            self.var_list.append(var)

        return var
