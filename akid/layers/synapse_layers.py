import abc
import sys
import inspect
import math

import tensorflow as tf

from ..core.blocks import ProcessingLayer
from ..core.common import (
    FILTER_WEIGHT_COLLECTION,
    AUXILLIARY_SUMMARY_COLLECTION,
    AUXILLIARY_STAT_COLLECTION
)
from ..core import initializers
from .. import backend as A
import helper_methods


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
                 in_channel_num,
                 out_channel_num,
                 initial_bias_value=0,
                 init_para={"name": "truncated_normal", "stddev": 0.1},
                 wd={"type": "l2", "scale": 5e-4},
                 max_norm=None,
                 do_stat_on_norm=False,
                 **kwargs):
        """
        Args:
            in_channel_num: int
                How many in channels this synapse layer has.
            out_channel_num: int
                How many out channels this synapse layer has.
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
        self.in_channel_num = in_channel_num
        self.out_channel_num = out_channel_num
        self.init_para = init_para
        self.wd = wd
        self.max_norm = max_norm
        self.do_stat_on_norm = do_stat_on_norm

        # Only do float conversion if not None.
        self.initial_bias_value = float(initial_bias_value) \
            if initial_bias_value is not None else None

    @abc.abstractmethod
    def _para_init(self):
        """
        Allocate learnable parameters.

        Every learnable layer should implement this method. It is made an
        abstract method here just to regularize the internal implementation of
        learning layers.

        Args:
            input: tensor
                The to be processed input in proper shape (it may be processed
                by the layer), will be passed to `_para_init` to do actual
                initialization.

        Return:
            None
        """
        raise NotImplementedError("Each learnable layer needs to implement"
                                  " this method to allocate and init"
                                  " parameters!")

    def _setup(self):
        self._para_init()

    def _post_setup(self):
        super(SynapseLayer, self)._post_setup()
        if self.do_stat_on_norm:
            for v in self.var_list:
                # Do not apply on biases.
                shape = v.get_shape().as_list()
                if len(shape) > 1:
                    _ = tf.reduce_sum(tf.square(v), range(0, len(shape)-1))
                    v_norms = tf.sqrt(_, name="l2norm")
                    tf.add_to_collection(AUXILLIARY_STAT_COLLECTION,
                                         v_norms)
                    tf.summary.histogram(
                        v.op.name + "l2norm",
                        v_norms,
                        collections=[AUXILLIARY_SUMMARY_COLLECTION])

    def _pre_forward(self, *args, **kwargs):
        super(SynapseLayer, self)._pre_forward(*args, **kwargs)

        if not self.initial_bias_value:
            self.log("Bias is disabled.")

    def on_para_update(self):
        if self.max_norm:
            self.log("Using max norm constrain of {}.".format(self.max_norm))
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
            self.clipped_filters = []

        return self.clipped_filters


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
                self.log("Using {} regularization with scale {}".format(
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
                    self.log.error("Type {} loss is not supported!".format(
                        self.wd["type"]))
                    sys.exit(1)
            except KeyError as e:
                raise Exception("`{}` not found in the provided regularization"
                                " parameters, `wd`. Perhaps you have some"
                                " typos.".format(e.message))

        return var, weight_decay

    def _get_initializer(self):
        if not self.init_para:
            init = initializers.get("default")
        else:
            name = self.init_para["name"]
            kwargs = self.init_para.copy()
            kwargs.pop("name")
            init = initializers.get(name, **kwargs)

        self.log("Weights of {} uses initializer {} with arguments {}".format(
            self.name, name, kwargs))

        return init


class ConvolutionLayer(SynapseLayer):
    def __init__(self, ksize, strides, padding, **kwargs):
        super(ConvolutionLayer, self).__init__(**kwargs)
        self.ksize = helper_methods.expand_kernel(ksize)
        self.strides = helper_methods.expand_kernel(strides)
        self.padding = padding
        self.ksize = ksize

    def _para_init(self):
        self.shape = [self.ksize[0], self.ksize[1],
                      self.in_channel_num, self.out_channel_num]
        self.weights, self._loss \
            = self._variable_with_weight_decay("weights", self.shape)

        if self.initial_bias_value is not None:
            self.biases = self._get_variable(
                'biases',
                [self.shape[-1]],
                initializer=tf.constant_initializer(self.initial_bias_value))

    def _pre_forward(self, input, *args, **kwargs):
        super(ConvolutionLayer, self)._pre_forward(*args, **kwargs)
        self.log("Padding method {}.".format(self.padding), debug=True)
        self.input_shape = input.get_shape().as_list()

    def _forward(self, input):
        conv = tf.nn.conv2d(input, self.weights, self.strides, self.padding)

        if self.initial_bias_value is not None:
            output = tf.nn.bias_add(conv, self.biases)
        else:
            output = conv

        self._data = output

    def backward(self, X_in):
        """
        According to the top-down inference results, reconstruct the input.
        """
        # Deconvolve.
        if self.initial_bias_value is not None:
            X_in = tf.nn.bias_add(X_in, -self.biases)

        self._data_g = tf.nn.conv2d_transpose(X_in,
                                              self.weights,
                                              self.input_shape,
                                              self.strides,
                                              self.padding,
                                              name="deconv")

        # TODO: Create reconstruction loss.

        return self.data_g


class SLUConvLayer(ConvolutionLayer):
    """
    Deprecated.

    This is a wrong idea. It tests sign after convolution, but it should be
    done before. Also, the sign test depends on the top down reconstruction
    value, which is a latent variable.

    Convolution layer preceded by Switchable Linear Unit, activation function
    developed by properly doing posterior inference in DRMM.
    """
    def _forward(self, X_in):
        depthwise = A.nn.depthwise_conv2d(X_in, self.weights, self.strides, self.padding)
        # Compute the sign of each pixel.
        sign = A.cast(depthwise > 0, A.float32)
        # Drop negative value.
        conv = depthwise * sign
        # Collapse dimensions.
        shape = conv.get_shape().as_list()
        shape.pop()
        shape.extend([self.out_channel_num, self.input_shape[-1]])
        conv = A.reshape(conv, shape=shape)
        conv = tf.reduce_sum(conv, axis=-1, name="slu_conv")

        if self.initial_bias_value is not None:
            output = tf.nn.bias_add(conv, self.biases, name="slu_conv_bias")
        else:
            output = conv

        self._data = output


class InnerProductLayer(SynapseLayer):
    def _forward(self, input):
        input = self._reshape(input)
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

        # Hold another addition info about the shape of input feature maps.
        self.in_shape = input_shape[1:]

        return reshaped_input

    def _para_init(self):
        self.shape = [self.in_channel_num, self.out_channel_num]
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

    def _forward(self, input):
        object_vector = self._preprocess(input)
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

    def _para_init(self):
        self.shape = [self.in_channel_num, self.out_channel_num]
        self.weights, self._loss = self._variable_with_weight_decay(
            'weights', self.shape)
        self.biases = self._get_variable(
            'biases',
            shape=[self.out_channel_num],
            initializer=tf.constant_initializer(0.0))


class RenderingMixtureLayer(SynapseLayer):
    """
    Implementation of *Ankit B. Patel et al A Probabilistic Framework for Deep
    Learning*.

    A rendering mixture layer is a *max-sum classifier* that passes the maximal
    log likelihood of all variations of a hidden class to the next layer. It is
    a generative model that first computes the posterior probability p(c | I),
    where c is the predicted class, and I is the input image (feature map),
    then uses the predicted class to reconstruct the input images. Formally,
    the process is a hard Expectation Maximization.
    """
    # Parameter init.
    # Reparameterize to weight and bias of neural network (TODO: check whether
    # it is necessary)
    # Reconstruct input
    # Compute reconstruction gradient
    pass


__all__ = [name for name, x in locals().items() if
           not inspect.ismodule(x) and not inspect.isabstract(x)]
