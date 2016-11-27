import abc
import sys
import inspect
import math

import tensorflow as tf

from ..utils import glog as log
from ..core.blocks import ProcessingLayer
from ..core.common import (
    SEED,
    FILTER_WEIGHT_COLLECTION,
    AUXILLIARY_SUMMARY_COLLECTION,
    AUXILLIARY_STAT_COLLECTION
)
from ..ops import msra_initializer


class SynapseLayer(ProcessingLayer):
    """
    An abstract class to model connections between neurons. It takes massive
    inputs, which are dendrites in neuroscience, and produces massive raw
    outputs, which are going to be further processed by consequent layers, such
    as pooling layers and non-linear activation layers. The outputs ultimately
    connect to next layer, which is axons in neuroscience. Convolution and
    Inner Product layers are two examples.

    A large number of connections needs to be initialized before training
    starts. Constructor parameter `initializer` lets user pass in the type of
    initializer one wants. Current support initializers are:

        * truncated_normal.
              Customizable parameters: stddev, standard deviation of the normal
              distribution; means are zero by default.
        * uniform
              Customizable parameters: range; A uniform initializer with range
              1 will have uniform distribution U(-1, 1).
    """
    def __init__(self,
                 out_channel_num,
                 initial_bias_value=0,
                 init_para={"name": "truncated_normal", "stddev": 0.1},
                 wd={"type": "l2", "scale": 5e-4},
                 max_norm=None,
                 do_stat_on_norm=False,
                 **kwargs):
        """
        Args:
            out_channel_num: an int
                How many channels this synapse layer has. This is a positional
                and required parameter.
            init_bias_value: a real number
                If bias is unwanted, set it to None explicitly.
            init_para: dict
                An dictionary that contains the name of the initializer to use,
                and its parameters. The name, and parameters have to exactly
                match with perdefined strings.
            wd: dict
                An dictionary that contains the `type` of regularization to use
                and its scale, which is to say the multiplier of the
                regularization loss term. If None, weight decay is not added.
            max_norm: a real number
                A real number that constrains the maximum norm of the filter of
                a channel could be at largest. If exceeding that value, it
                would be projected back.
            do_stat_on_norm: boolean
                Whether to collect statistics on norms of filters. If true,
                summary ops that calculate norms will be added to
                `AUXILIARY_SUMMARY_COLLECTION`, which could be used further. It
                is mainly used for quantitatively evaluate how many filters are
                dead during training.
        """
        super(SynapseLayer, self).__init__(**kwargs)
        self.out_channel_num = out_channel_num
        self.init_para = init_para
        self.wd = wd
        self.max_norm = max_norm
        self.do_stat_on_norm = do_stat_on_norm

        # Only do float conversion if not None.
        self.initial_bias_value = float(initial_bias_value) \
            if initial_bias_value is not None else None

    @abc.abstractmethod
    def _para_init(self, input):
        """
        Allocate learnable parameters, do stat and visualizations logistics.

        Every learnable layer should implement this method. It is made an
        abstract method here just to regularize the internal implementation of
        learning layers.

        Args:
            input: tensor
                The to be processed input in proper shape(it may be processed
                by the layer), will be passed to `_para_init` to do actual
                initialization.

        Return:
            None
        """
        raise NotImplementedError("Each learnable layer needs to implement"
                                  " this method to allocate and init"
                                  " parameters!")
        sys.exit()

    def _pre_setup(self, *args, **kwargs):
        super(SynapseLayer, self)._pre_setup(*args, **kwargs)

        if not self.initial_bias_value:
            log.info("Bias is disabled.")

    def _post_setup_shared(self):
        super(SynapseLayer, self)._post_setup_shared()
        if self.max_norm:
            log.info("Using max norm constrain of {}.".format(self.max_norm))
            # Create the op to apply max norm clip on weights.
            self.clipped_filters = []
            for v in self.var_list:
                # Do not apply on biases.
                shape = v.get_shape().as_list()
                if len(shape) > 1:
                    out_channel_dim = len(shape) - 1
                    filter_tuple = tf.split(out_channel_dim, shape[-1], v)
                    clipped_filter_list = []
                    for f in filter_tuple:
                        clipped_filter_list.append(
                            tf.clip_by_norm(f, self.max_norm))
                    clipped_v = tf.concat(out_channel_dim,
                                          clipped_filter_list)
                    self.clipped_filters.append(tf.assign(v, clipped_v))
        else:
            self.clipped_filters = None

        if self.do_stat_on_norm:
            for v in self.var_list:
                # Do not apply on biases.
                shape = v.get_shape().as_list()
                if len(shape) > 1:
                    out_channel_dim = len(shape) - 1
                    _ = tf.reduce_sum(tf.square(v), range(0, len(shape)-1))
                    v_norms = tf.sqrt(_, name="l2norm")
                    tf.add_to_collection(AUXILLIARY_STAT_COLLECTION,
                                         v_norms)
                    tf.histogram_summary(
                        v.op.name + "l2norm",
                        v_norms,
                        collections=[AUXILLIARY_SUMMARY_COLLECTION])

    def _variable_with_weight_decay(self, name, shape):
        """Helper to create an initialized Variable with weight decay.

        Args:
            name: name of the variable
            shape: list of ints

        Returns:
            (Variable Tensor, Weight Decay Loss) If `self.wd` is `None`, then
            the returned weight decay loss would be `None`.
        """
        var = self._get_variable(name, shape, self._get_initializer())
        if len(shape) > 1:
            # Add non-bias filters to the collection.
            tf.add_to_collection(FILTER_WEIGHT_COLLECTION, var)

        weight_decay = None
        if self.wd and self.wd["scale"] is not 0:
            try:
                log.info("Using {} regularization with scale {}".format(
                    self.wd["type"], self.wd["scale"]))

                if self.wd["type"] == "l2":
                    weight_decay = tf.mul(tf.nn.l2_loss(var),
                                          self.wd["scale"],
                                          name=name + '/l2_loss')
                elif self.wd["type"] == "l1":
                    weight_decay = tf.mul(tf.reduce_sum(tf.abs(var)),
                                          self.wd["scale"],
                                          name=name + '/l1_loss')
                else:
                    log.error("Type {} loss is not supported!".format(
                        self.wd["type"]))
                    sys.exit(1)
            except KeyError as e:
                log.error("`{}` not found in the provided regularization"
                          " parameters, `wd`. Perhaps you have some"
                          " typos.".format(e.message))
                sys.exit(1)

        return var, weight_decay

    def _get_default_initializer(self):
        # By default, we use the most preliminary initialization (for
        # conforming with torch).
        log.info("Weights of {} uses default initialization.".format(
            self.name))
        # The strange factor here is to make variance `1/sqrt(dim)`. For
        # the meaning of `dim`, see the doc of
        # `tf.uniform_unit_scaling_initializer`.
        return tf.uniform_unit_scaling_initializer(factor=1.0/(3)**0.5,
                                                   seed=SEED)

    def _get_initializer(self):
        if not self.init_para:
            return self._get_default_initializer()
        else:
            try:
                name = self.init_para["name"]
                if name is "default":
                    return self._get_default_initializer()
                elif name is "truncated_normal":
                    log.info("Weights of {} uses truncated normal initializer"
                             " with stddev {}".format(
                                 self.name, self.init_para["stddev"]))
                    return tf.truncated_normal_initializer(
                        stddev=self.init_para["stddev"],
                        seed=SEED)
                elif name is "uniform":
                    range = self.init_para["range"]
                    log.info("Weights of {} uses uniform initializer with"
                             " stddev {}".format(self.name, range))
                    return tf.random_uniform_initializer(minval=-range,
                                                         maxval=range,
                                                         seed=SEED)
                elif name is "uniform_unit_scaling":
                    try:
                        factor = self.init_para["factor"]
                    except KeyError as e:
                        log.info("Key factor is not found in `init_para`. Use"
                                 " 1")
                        factor = 1
                    log.info("Weights of {} uses uniform unit scaling"
                             " initializer of factor {}".format(
                                 self.name, factor))
                    return tf.uniform_unit_scaling_initializer(factor=factor,
                                                               seed=SEED)
                elif name is "msra_init":
                    try:
                        factor = self.init_para["factor"]
                    except KeyError as e:
                        log.info("Key factor is not found in `init_para`. Use"
                                 " 1")
                        factor = 1
                    log.info("Weights of {} uses unit gradient (msra"
                             " initializer) initializer with factor {}".format(
                                 self.name, factor))
                    return msra_initializer(factor=factor, seed=SEED)
                else:
                    log.error("{} is not supported!".format(name))
                    sys.exit(0)
            except KeyError as e:
                log.error("`{}` not found in the provided initialization"
                          " parameters, `init_para`. Perhaps you have some"
                          " typos.".format(e.message))
                sys.exit(1)


class ConvolutionLayer(SynapseLayer):
    def __init__(self, ksize, strides, padding, **kwargs):
        super(ConvolutionLayer, self).__init__(**kwargs)
        self.strides = strides
        self.padding = padding
        self.ksize = ksize

    def _para_init(self, input):
        input_shape = input.get_shape().as_list()
        self.shape = [self.ksize[0], self.ksize[1],
                      input_shape[-1], self.out_channel_num]
        self.weights, self._loss \
            = self._variable_with_weight_decay("weights", self.shape)

        if self.initial_bias_value is not None:
            self.biases = self._get_variable(
                'biases',
                [self.shape[-1]],
                initializer=tf.constant_initializer(self.initial_bias_value))

    def _setup(self, input):
        self._para_init(input)

        log.debug("Padding method {}.".format(self.padding))
        conv = tf.nn.conv2d(input, self.weights, self.strides, self.padding)

        if self.initial_bias_value is not None:
            output = tf.nn.bias_add(conv, self.biases)
        else:
            output = conv

        self._data = output


class InnerProductLayer(SynapseLayer):
    def _setup(self, input):
        input = self._reshape(input)
        self._para_init(input)

        ip = tf.matmul(input, self.weights)
        if self.initial_bias_value is not None:
            ip = tf.nn.bias_add(ip, self.biases)

        self._data = ip

    def _reshape(self, input):
        """
        Flatten all input feature maps.

        Args:
            input: tensor
                The processed input from previous layer, or just input.
        Return:
            input: tensor
                The reshaped input that could be processed by this layer.
        """
        input_shape = input.get_shape().as_list()
        in_channel_num = input_shape[1]
        # Check the input shape, if it is not 2D tensor, reshape all remaining
        # dimensions into one.
        reshaped_input = input
        if len(input_shape) != 2:
            in_channel_num = 1
            for i in input_shape[1:]:
                in_channel_num *= i
            flattened = tf.reshape(input, [input_shape[0], in_channel_num])
            reshaped_input = flattened

        self.shape = [in_channel_num, self.out_channel_num]

        # Hold another addition info about the shape of input feature maps.
        self.in_shape = input_shape[1:]

        return reshaped_input

    def _para_init(self, input):
        self.weights, self._loss = self._variable_with_weight_decay(
            'weights', self.shape)
        if self.initial_bias_value is not None:
            self.biases = self._get_variable(
                'biases',
                shape=[self.out_channel_num],
                initializer=tf.constant_initializer(self.initial_bias_value))


class InvariantInnerProductLayer(SynapseLayer):
    """
    This is a half-baked idea. See blog note for details.
    """
    NAME = "Invariant_IP"

    def _setup(self, input):
        object_vector = self._preprocess(input)
        self._para_init(object_vector)
        ip = tf.matmul(object_vector, self.weights)
        ip_plus_bias = tf.nn.bias_add(ip,
                                      self.biases,
                                      name=InvariantInnerProductLayer.NAME)

        self._data = ip_plus_bias

    def _preprocess(self, input):
        # Gather some info.
        input_shape = input.get_shape().as_list()
        batch_size = input_shape[0]
        fmap_h = input_shape[1]
        fmap_w = input_shape[2]
        in_channel_num = input_shape[3]
        # Do object existence summary.
        object_vector = tf.nn.max_pool(input,
                                       [1, fmap_h, fmap_w, 1],
                                       [1, fmap_h, fmap_w, 1],
                                       "VALID")
        # Reshape input to remove the one dim axes.
        object_vector = tf.reshape(object_vector, [batch_size, in_channel_num])
        return object_vector

    def _para_init(self, input):
        self.shape = [input.get_shape().as_list()[1], self.out_channel_num]
        self.weights, self._loss = self._variable_with_weight_decay(
            'weights', self.shape)
        self.biases = self._get_variable(
            'biases',
            shape=[self.out_channel_num],
            initializer=tf.constant_initializer(0.0))


__all__ = [name for name, x in locals().items() if
           not inspect.ismodule(x) and not inspect.isabstract(x)]
