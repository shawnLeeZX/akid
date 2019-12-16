"""
`akid` builds another layer of abstraction on top of *Tensor*: *Block*.
Tensor can be taken as the media/formalism signal propagates in digital world,
while Block is the data processing entity that processes inputs and emits
outputs.

Best designs mimic nature. `akid` tries to reproduce how signals in nature
propagates. Information flow can be abstracted as data propagating through
inter-connected blocks, each of which processes inputs and emits outputs. For
example, a vision classification system is a block that takes image inputs and
gives classification results. Everything is a `Block` in `akid`.

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
from deprecation import deprecated

import tensorflow as tf

from ..utils import glog as log
from .common import (
    TRAIN_SUMMARY_COLLECTION,
    VALID_SUMMARY_COLLECTION,
    SPARSITY_SUMMARY_SUFFIX,
    FILTER_WEIGHT_COLLECTION,
)
from ..core import initializers
from .. import backend as A
from .interface_blocks import UpdateBlock
import six


class Block(six.with_metaclass(abc.ABCMeta, object)):
    """
    The top level class. Everything should be its sub-class.

    A `Block` holds computational graph. Sub-class should implement `_setup` to
    build the graph. It does not enforce how the computational graph should be
    used, which should be the responsibility of sub classes.

    Any outputs of a block is of type Tensor. Note no builtin tensor type
    exists. Depending on the backend, the Tensor is the type Tensor from the
    backend. Such a requirement is to normalize data type in outputs.

    `Block` supports add arbitrary code after setup by offering
    `post_setup_hook`. All functions added in the hook will be called after
    setup. `self` is passed as an argument to functions in the hook.

    Call `setup` of each block before using it.
    """
    def __init__(self, name=None, debug=False, **kwargs):
        super(Block, self).__init__(**kwargs)

        A.inc_block_count(self.__class__)

        if name is None:
            self.name = self.get_default_name()
        else:
            self.name = name
        self.debug = debug

        # Hooks that are called after setup.
        self.post_setup_hook = []
        self.pre_setup_hook = []

        self.is_setup = None

        # Enable logging to stderr for each individual, so logging is available
        # when not using the Kid.
        log.init(akid_logger=True)

    def get_default_name(self):
        count = A.get_block_count(self.__class__)
        return "{}_{}".format(self.__class__.NAME, count)

    def setup(self):
        with A.variable_scope(self.name):
            self._pre_setup()
            self._setup()
            self._post_setup()

        self.is_setup = True

    def _setup(self):
        pass

    def teardown(self):
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

    def get_clone(self):
        """
        To clone is to create a copy of this block that has an independent copy
        of each component of this block. The relationship between get_clone and
        get_copy is like the that between deepcopy and copy.
        """
        return self.get_copy()


class DataBlock(Block):
    """
    A block that has a property `data` to supply data. Data does not
    necessarily flow in, or are operated in batches in this block.
    """
    @abc.abstractmethod
    def data(self):
        """
        An abstract method to enforce all sub-classes to provide their
        processed data through this interface.
        """
        raise NotImplementedError("Each concrete block needs to implement this"
                                  " method to provide an interface to offer"
                                  " data!")


class FlowBlock(DataBlock):
    """
    Abstract class for naive data flow. It implements naive data flow
    interfaces without any other functionalities.
    """
    def forward(self, *args, **kwargs):
        """
        Args:
            All arguments will be passed to the actual `_forward` function.
        """
        if not self.is_setup:
            self.setup()

        with A.variable_scope(self.name):
            out = self._forward(*args, **kwargs)

        return out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ProcessingBlock(FlowBlock):
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
    implemented in `_forward` method. The return value of `_forward` will be
    returned by `forward`. To add arbitrary code before and after `_forward`,
    use `pre_forward_hook`; to add after, use `post_forward_hook`. All
    arguments passed to `_forward` are also available in functions in the
    hooks.
    """
    _flags = ["do_summary", "summarize_output", "is_mon", "done_first_batch_monitoring_pass"]

    def __init__(self,
                 do_summary=None,
                 bag=None,
                 summarize_output=None,
                 **kwargs):
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
            summarize_output: bool
                Summarize on output, including data, loss etc. If true, a name
                tag will be given to the outputs, consequently, summarizing
                events will be created for tensorboard. This semantics is
                enforced by each individual layer, by implementing properly.
        """
        super(ProcessingBlock, self).__init__(**kwargs)

        self.do_summary = do_summary
        self.log("{} has bag: {}".format(self.name, bag))
        self.bag = bag
        self.summarize_output = summarize_output

        # Hooks
        self.pre_forward_hook = []
        self.post_forward_hook = []

        # Some operations only run at the first forward pass, so set a flag.
        self.done_first_pass = False

        self.done_first_batch_monitoring_pass = False
        # Whether the brain is in the batch monitoring mode.
        self.is_mon = False

    def set_flag(self, flag_name, v):
        if flag_name in self.__class__._flags:
            setattr(self, flag_name, v)
        else:
            raise ValueError("{} does not have flag {}".format(flag_name))

    def set_do_summary_flag(self, v):
        self.do_summary = v

    def summarize_data(self, data, val=False, collections=None):
        """
        Helper function to summarize data with Tensorboard. `data` needs to be
        numerical, e.g., a Tensor, or a numpy array.

        Args:
            val: bool
                Whether to summarize data in val mode as well. If False,
                Summary ops won't be created for data in val mode won.
        """
        if not self.do_summary:
            return

        if not self.done_first_pass\
           or val and self.is_val and not self.done_first_pass_val:
            if collections is None:
                if not val:
                    collections = [TRAIN_SUMMARY_COLLECTION]
                else:
                    collections = [VALID_SUMMARY_COLLECTION]

            self._data_summary(data, collections=collections)

    def forward(self, *args, **kwargs):
        """
        Args:
            All arguments will be passed to the actual `_forward` function.
        """
        if not self.is_setup:
            self.setup()

        with A.variable_scope(self.name):
            self._pre_forward(*args, **kwargs)
            out = self._forward(*args, **kwargs)
            post_out = self._post_forward(*args, **kwargs)
            if not self.done_first_pass:
                self.done_first_pass = True

        if post_out is None:
            return out
        else:
            return post_out

    def _pre_forward(self, *args, **kwargs):
        for f in self.pre_forward_hook:
            f(self, *args, **kwargs)

    def _post_forward(self, *args, **kwargs):
        for f in self.post_forward_hook:
            f(self, *args, **kwargs)

        if self.done_first_pass or not self.do_summary:
            return

        if self.data is not None:
            self._data_summary(self.data, sparsity_summary=True, collections=[TRAIN_SUMMARY_COLLECTION])

    @abc.abstractmethod
    def _forward(self):
        """
        An abstract method that must be overrided.
        """
        raise NotImplementedError('Each sub-layer needs to implement this'
                                  'method to process data!')

    def _data_summary(self, data, sparsity_summary=False, collections=None):
        """
        Helper function to do statistical summary on the bundle of data.
        """
        if type(data) is not list and type(data) is not tuple:
            data = [data]

        for d in data:
            name = A.get_name(d)
            if name:
                if not A.is_numerical(d):
                    return

                self.log("Do tensorboard summary on outputs {} of {}".format(
                    name, self.name))

                shape = A.get_shape(d)
                if shape != 0:
                    dim = len(shape)
                else:
                    dim = 0
                if dim == 0:
                    A.summary.scalar(name, d, collections=collections)
                else:
                    A.summary.histogram(name,
                                        d,
                                        collections=collections)
                    if sparsity_summary:
                        A.summary.scalar(
                            A.append_suffix(name, SPARSITY_SUMMARY_SUFFIX),
                            A.nn.zero_fraction(d,
                                               # The scope removing does
                                               # nothing for tensorflow
                                               # backend, since the name gotten
                                               # is already scope removed.
                                               name=A.remove_scope_from_name(name) \
                                               + '/' \
                                               + SPARSITY_SUMMARY_SUFFIX),
                            collections=collections)


class ValidatableProcessingBlock(ProcessingBlock):
    """
    Since normally we have the two types of data, a.k.a, validation data and
    training data, `ValidatableProcessingBlock` support blocks to operate in
    two modes.
    """
    _flags = copy.copy(ProcessingBlock._flags)
    _flags.append("do_summary_on_val")

    def __init__(self, do_summary_on_val=None, **kwargs):
        """
        Args:
            do_summary_on_val: bool
                Whether to do summary on validation copy.
        """
        super(ValidatableProcessingBlock, self).__init__(**kwargs)

        # A Boolean flag to indicate whether this block is in validation mode.
        self.mode = "train"
        self.done_first_pass_val = False
        self.do_summary_on_val = do_summary_on_val

    @property
    def is_val(self):
        return self.mode == A.Mode.VAL

    def set_do_summary_on_val_flag(self, v):
        self.do_summary_on_val = v

    def forward(self, *args, **kwargs):
        out = super(ValidatableProcessingBlock, self).forward(*args, **kwargs)

        if not self.done_first_pass_val and self.is_val:
            self.done_first_pass_val = True

        return out

    def _pre_setup(self):
        super(ValidatableProcessingBlock, self)._pre_setup()

        if self.is_val:
            A.get_variable_scope().reuse_variables()

    def get_val_copy(self):
        """
        Get a copy for validation.

        Since a processing layer is learned, it has to be taken out for
        evaluation from time to time.
        """
        if A.backend() == A.TF:
            val_copy = self.get_copy()
            val_copy.name = self.name + '_val'
        elif A.backend() == A.TORCH:
            # Since torch is dynamic graph, no need to create a copy for
            # validation.
            val_copy = self
        else:
            raise ValueError("Not supported backend.")

        val_copy.set_val()
        return val_copy

    def set_val(self, val):
        self.mode = A.Mode.VAL if val else A.Mode.TRAIN

    def _post_forward(self, *args, **kwargs):
        for f in self.post_forward_hook:
            f(self, *args, **kwargs)

        # If has done forward for building computational graphs that exist in
        # form of ops, return. But it is possible we still need to build ops
        # for validation block.
        if self.done_first_pass and not self.is_val:
            return

        if not self.do_summary:
            return

        if not self.is_val\
           or (self.is_val\
               and self.do_summary_on_val\
               and not self.done_first_pass_val):

            collection = VALID_SUMMARY_COLLECTION if self.is_val \
                else TRAIN_SUMMARY_COLLECTION

            if self.data is not None:
                self._data_summary(self.data, sparsity_summary=True, collections=[collection])


class ShadowableBlock(ValidatableProcessingBlock):
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

    def setup(self):
        """
        For shadow copies, only broadcast parameters need to be retrieved
        again, so only `_setup` will be called.
        """
        if self.is_shadow:
            with A.variable_scope(self.name):
                self._setup()
        else:
            super(ShadowableBlock, self).setup()

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


class ProcessingLayer(GenerativeBlock, UpdateBlock):
    """
    An abstract layer for data processing layer in the brain.

    A `ProcessingLayer` is nothing if it does not process data. So every
    sub-class of `ProcessingLayer` should possess a `data` property as
    interface to provide the data it processes.

    For now, all processing layers only have one output, and provide it via
    property `data`, which can be a list to hold multiple data. So it overrides
    the `data` method of `ProcessingBlock`.

    Optionally a `ProcessingLayer` could have a `loss` property for loss (or
    losses) in this layer and a `eval` property for any evaluation metrics in
    this layer. `loss` should be a list if there are multiple losses, so is
    eval graphs.

    If `inputs` in the constructor is not None (in this case it should be a
    list), this layer is supposed to have multiple inputs. Refer to
    `system.GraphSystem` for more explanation.
    """
    _flags = copy.copy(ValidatableProcessingBlock._flags)
    _flags.append("summarize_variables")

    def __init__(self,
                 moving_average_decay=None,
                 inputs=None,
                 summarize_variables=None,
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
            summarize_variables: bool
                Whether to do tensorboard summary on variables.
        """
        super(ProcessingLayer, self).__init__(**kwargs)

        assert moving_average_decay is None or \
            moving_average_decay >= 0.5 and moving_average_decay < 1, \
            ("Invalid moving_average_decay value {}. Should be None or"
             " between [0.5, 1]".format(moving_average_decay))
        self.moving_average_decay = moving_average_decay

        self.inputs = inputs

        # Bookkeeping all variables.
        self.summarize_variables = summarize_variables
        self.var_list = []

    def set_shadow(self):
        super(ProcessingLayer, self).set_shadow()
        self.var_list = []

    @deprecated(details="This method only worked for tensorflow backends, and is broken now.")
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
            name = "default"
            init = initializers.get(name)
            kwargs = {}
        else:
            name = init_para["name"]
            kwargs = init_para.copy()
            kwargs.pop("name")
            init = initializers.get(name, **kwargs)

        if not self.is_setup:
            self.log("Variables of {} will use initializer {} with arguments {}"
                     " if the variables have not existed yet".format(
                         self.name, name, kwargs))

        return init

    def _pre_setup(self):
        super(ProcessingLayer, self)._pre_setup()

        if self.moving_average_decay:
            # We pass current training step to moving average to speed up
            # updates moving average of variables at the beginning of the
            # training since moving average is useful only later.
            step = A.get_step()
            self.moving_averages = tf.train.ExponentialMovingAverage(
                self.moving_average_decay, step)

    @property
    def outputs(self):
        """
        Return all outputs of this block in a list.

        Outputs that are accessed thorough functions call will not be included.
        """
        outputs = []
        if self.data:
            outputs.extend(self.data) if type(self.data) is list else outputs.append(self.data)
        if self.eval:
            outputs.extend(self.eval)
        if not callable(self.loss):
            if self.loss:
                outputs.extend(self.loss) if type(self.loss) is list else outputs.append(self.loss)

        return outputs

    def _post_setup(self):
        super(ProcessingLayer, self)._post_setup()
        # Maintain moving averages of variables.
        if self.moving_average_decay and len(self.var_list) is not 0:
            self.moving_averages_op = self.moving_averages.apply(self.var_list)

            for var in self.var_list:
                var_average = self.moving_averages.average(var)
                self._var_summary(A.get_name(var) + "_average", var_average)

        if self.do_summary and self.summarize_variables:
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
        self.log("This layer has {} parameters.".format(total_para_num))

    def _post_forward(self, *args, **kwargs):
        # NOTE: this method does not call that of its super class. It has code
        # redundancy, and may be improved.
        for f in self.post_forward_hook:
            f(self, *args, **kwargs)

        # If has done forward for building computational graphs that exist in
        # form of ops, return. But it is possible we still need to build ops
        # for validation block.

        # Do not do summary for shadow copies.
        if self.is_shadow and not self.debug:
            return

        if self.done_first_pass and not self.is_val:
            return

        if not self.do_summary:
            return

        if not self.is_val\
           or (self.is_val\
               and self.do_summary_on_val\
               and not self.done_first_pass_val):
            if self.is_val:
                collections = [VALID_SUMMARY_COLLECTION]
            else:
                collections = [TRAIN_SUMMARY_COLLECTION]

            if self.summarize_output and self.data is not None:
                self._data_summary(self.data, collections=collections)
            if self.loss is not None:
                self._data_summary(self.loss, collections=collections)
            if self.eval is not None:
                self._data_summary(self.eval, collections=collections)


    def _on_update(self, K_prev):
        # Keep propagating the Riemannian metric.
        self.K = K_prev

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
        self.log("Do tensorboard summary on variables {} of {}".format(
            A.get_name(var), self.name))
        if len(A.get_shape(var)) is 0:
            A.summary.scalar(tag,
                             var,
                             collections=[TRAIN_SUMMARY_COLLECTION])
        else:
            A.summary.histogram(tag,
                                var,
                                summary_on_grad=True,
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
    def verbose_eval(self):
        """
        More verbosely evaluation results. Put the results under
        `_verbose_eval`.
        """
        if hasattr(self, "_verbose_eval"):
            return self._verbose_eval
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
        Allocate or retrieve a trainable variable.

        The following moving average only works in tensorflow backend.  If the
        variable has already existed, depending on `moving_average_decay`'s
        value, moving average of it(when `moving_average_decay` has a value) or
        the original variable(when `moving_average_decay` is None) would be
        returned.
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
                self.log("Reuse paras {}".format(A.get_name(var)))
        else:
            # Append it to the var list, do moving average later in
            # `_post_setup`.
            self.var_list.append(var)

        return var
