from __future__ import division

from __future__ import absolute_import
import abc
import inspect

import tensorflow as tf
from deprecated import deprecated

from ..core.blocks import ProcessingLayer
from ..core.common import (
    AUXILLIARY_SUMMARY_COLLECTION,
    AUXILLIARY_STAT_COLLECTION
)
from .. import backend as A
from .activation_layers import BatchNormalizationLayer
from .. import nn

from torch.nn import BatchNorm2d
from six.moves import range


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
                 init_para={"name": "default"},
                 max_norm=None,
                 do_stat_on_norm=False,
                 # wd={"type": "l2", "scale": 5e-4},  # Save for future reference.
                 wd=None,
                 wd_on_bias=False,
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
            wd: dict
                An dictionary that contains the `type` of regularization to use
                and its scale, which is to say the multiplier of the
                regularization loss term. If None, weight decay is not added.
            wd_on_bias: boolean
                Whether to apply weight decay on biases.
        """
        super(SynapseLayer, self).__init__(**kwargs)
        self.in_channel_num = in_channel_num
        self.out_channel_num = out_channel_num
        self.init_para = init_para
        self.max_norm = max_norm
        self.do_stat_on_norm = do_stat_on_norm
        self.wd = wd
        self.wd_on_bias = wd_on_bias

        # Only do float conversion if not None.
        self.initial_bias_value = float(initial_bias_value) \
            if initial_bias_value is not None else None

    def _post_setup(self):
        super(SynapseLayer, self)._post_setup()
        if self.weights is not None:
            self.log("Weight shape {}".format(A.get_shape(self.weights)))

        if self.initial_bias_value is None:
            self.log("Bias is disabled.")
        else:
            # It is possible some layers does not use bias.
            if self.biases is not None:
                self.log("Bias shape {}".format(A.get_shape(self.biases)))

        if self.wd is not None:
            self.log("Using {} regularization with scale {}".format(
                self.wd["type"], self.wd["scale"]))

            if self.wd_on_bias:
                self.log("Apply regularizer {} with scale {} on bias.".format(
                    self.wd["type"], self.wd["scale"]))


        if self.do_stat_on_norm:
            for v in self.var_list:
                # Do not apply on biases.
                shape = v.get_shape().as_list()
                if len(shape) > 1:
                    _ = tf.reduce_sum(tf.square(v), list(range(0, len(shape)-1)))
                    v_norms = tf.sqrt(_, name="l2norm")
                    tf.add_to_collection(AUXILLIARY_STAT_COLLECTION,
                                         v_norms)
                    tf.summary.histogram(
                        v.op.name + "l2norm",
                        v_norms,
                        collections=[AUXILLIARY_SUMMARY_COLLECTION])

    def _pre_forward(self, *args, **kwargs):
        super(SynapseLayer, self)._pre_forward(*args, **kwargs)
        if self.wd and self.wd["scale"] is not 0:
            self._loss = nn.regularizers.compute(self.wd["type"],
                                                 var=self.weights,
                                                 scale=self.wd["scale"],
                                                 name='weight_loss' if self.summarize_output else None)

        if self.initial_bias_value is not None:
            if self.wd_on_bias and self.wd["scale"] is not 0:
                loss = nn.regularizers.compute(self.wd["type"],
                                               var=self.biases,
                                               scale=self.wd["scale"],
                                               name='bias_loss' if self.summarize_output else None)

                if self._loss is not None:
                    self._loss += loss
                else:
                    self._loss = loss


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
                    filter_tuple = tf.split(axis=out_channel_dim, num_or_size_splits=shape[-1], value=v)
                    clipped_filter_list = []
                    for f in filter_tuple:
                        clipped_filter_list.append(
                            tf.clip_by_norm(f, self.max_norm))
                    clipped_v = tf.concat(axis=out_channel_dim,
                                          values=clipped_filter_list)
                    self.clipped_filters.append(tf.assign(v, clipped_v))
        else:
            self.clipped_filters = []

        return self.clipped_filters

    def get_clone(self):
        c = super(SynapseLayer, self).get_copy()

        # TODO: current clone after setup does not have names.
        if self.is_setup:
            c.weights = A.Tensor(A.value(c.weights), requires_grad=True)
            c.var_list = [c.weights]
            if c.biases is not None:
                c.biases = A.Tensor(A.value(c.biases), requires_grad=True)
                c.var_list.append(c.biases)

        return c

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, v):
        self._weights = v

    @property
    def biases(self):
        return self._biases

    @biases.setter
    def biases(self, v):
        self._biases = v


class ConvolutionLayer(SynapseLayer):
    NAME = "Conv2D"
    def __init__(self,
                 ksize,
                 strides=1,
                 padding="SAME",
                 depthwise=False,
                 bn=False,
                 **kwargs):
        super(ConvolutionLayer, self).__init__(**kwargs)
        self.ksize = [ksize, ksize] if type(ksize) is int else ksize

        self.strides = strides
        if A.backend() == A.TORCH:
            if type(strides) is int:
                self.strides = [strides, strides]
            elif len(strides) == 4:
                # The stride format is of that of tensorflow format. Extract the
                # stride to torch format.
                self.strides = [strides[1], strides[2]]

        self.padding = padding
        self.depthwise = depthwise
        self.bn = bn

    def get_weigth_shape(self):
        if A.backend() == A.TORCH:
            if len(self.ksize) == 4:
                # The ksize format is of that of tensorflow format. Extract the
                # ksize to torch format.
                self.ksize = [self.ksize[1], self.ksize[2]]
            shape = [self.out_channel_num, self.in_channel_num,
                          self.ksize[0], self.ksize[1]]
        else:
            shape = [self.ksize[0], self.ksize[1],
                          self.in_channel_num, self.out_channel_num]

        return shape

    def _setup(self):
        self.shape = self.get_weigth_shape()
        self._weights = self._get_variable("weights",
                                          self.shape,
                                          self._get_initializer(self.init_para))

        if self.initial_bias_value is not None:
            self._biases = self._get_variable(
                'biases',
                [self.out_channel_num if not self.depthwise else self.out_channel_num * self.in_channel_num],
                initializer=self._get_initializer(init_para={"name": "constant",
                                                             "value": self.initial_bias_value}))
        else:
            self._biases = None

        self.log("Padding method {}.".format(self.padding), debug=True)

        # Emergency code. Should be cleansed in the future.
        if self.bn:
            self.bn = BatchNorm2d(self.out_channel_num, affine=False)
            self.bn = self.bn.cuda() if A.use_cuda() else self.bn

    def _pre_forward(self, input, *args, **kwargs):
        super(ConvolutionLayer, self)._pre_forward(*args, **kwargs)
        self.input_shape = A.get_shape(input)

    def _forward(self, input):
        name = 'fmap' if self.summarize_output else None

        if self.depthwise:
            conv = A.nn.depthwise_conv2d(input,
                                         self.weights,
                                         self.biases,
                                         self.strides,
                                         self.padding,
                                         name=name)
        else:
            conv = A.nn.conv2d(input, self.weights,
                               self.biases, self.strides,
                               self.padding, name=name)

        self._data = conv

        # Emergency code. Should be cleansed in the future.
        if self.bn:
            self._data = self.bn(self._data)

        return self._data

    @deprecated(reason="Legacy code.")
    def _on_update_stub(self, K_prev):
        # Save the metric for this RKHS
        self.metric = K_prev
        # Compute the metric for the next layer

        if self.bn:
            W = A.nn.normalize_weight(self.weights)
        else:
            W = self.weights
        self.K = A.nn.nn_riemannic_metric(K_prev, W, self.biases if not self.bn else None)

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


class SelfNormalizedConvolutionLayer(SynapseLayer):
    NAME = "SNConv2D"
    def __init__(self,
                 ksize,
                 strides=1,
                 padding="SAME",
                 depthwise=False,
                 **kwargs):
        super(SelfNormalizedConvolutionLayer, self).__init__(**kwargs)
        self.ksize = [ksize, ksize] if type(ksize) is int else ksize

        self.strides = strides
        if A.backend() == A.TORCH:
            if len(strides) == 4:
                # The stride format is of that of tensorflow format. Extract the
                # stride to torch format.
                self.strides = [strides[1], strides[2]]

        self.padding = padding
        self.depthwise = depthwise

    def get_weigth_shape(self):
        if A.backend() == A.TORCH:
            if len(self.ksize) == 4:
                # The ksize format is of that of tensorflow format. Extract the
                # ksize to torch format.
                self.ksize = [self.ksize[1], self.ksize[2]]
            shape = [self.out_channel_num, self.in_channel_num,
                          self.ksize[0], self.ksize[1]]
        else:
            shape = [self.ksize[0], self.ksize[1],
                          self.in_channel_num, self.out_channel_num]

        return shape

    def _setup(self):
        self.shape = self.get_weigth_shape()
        self._weights = self._get_variable("weights",
                                          self.shape,
                                          self._get_initializer(self.init_para))

        if self.initial_bias_value is not None:
            self._biases = self._get_variable(
                'biases',
                [self.out_channel_num if not self.depthwise else self.out_channel_num * self.in_channel_num],
                initializer=self._get_initializer(init_para={"name": "constant",
                                                             "value": self.initial_bias_value}))
            if A.DATA_FORMAT == "CHW":
                self.biases = A.reshape(self.biases, [1, -1, 1, 1])
            elif A.DATA_FORMAT == "HWC":
                self.biases = A.reshape(self.biases, [1, 1, 1, -1])
            else:
                assert False, "Should not reach here. Data format is wrong."
        else:
            self._biases = None

        self.log("Padding method {}.".format(self.padding), debug=True)

    def _pre_forward(self, input, *args, **kwargs):
        super(SelfNormalizedConvolutionLayer, self)._pre_forward(*args, **kwargs)
        self.input_shape = A.get_shape(input)

    def _forward(self, input):
        name = 'fmap' if self.summarize_output else None

        conv = A.nn.conv2d(input, self.weights,
                           None, self.strides,
                           self.padding, name=name)

        self.mean = A.mean(conv, dim=[0, 2, 3], keep_dim=True)
        self.std = A.std(conv, dim=[0, 2, 3], keep_dim=True)
        out = (conv - self.mean) / self.std

        out = out + self.biases

        self._data = out

        return self._data


SNConv2D = SelfNormalizedConvolutionLayer


class Convolution1DLayer(SynapseLayer):
    NAME = "Conv1D"

    def __init__(self, ksize, stride, padding, **kwargs):
        super(Convolution1DLayer, self).__init__(**kwargs)
        self.ksize = ksize
        self.stride = stride
        self.padding = padding

    def _setup(self):
        if A.DATA_FORMAT == "CHW":
            self.shape = [self.out_channel_num, self.in_channel_num, self.ksize]
        else:
            raise ValueError("Data format {} is not supported yet.".format(A.DATA_FORMAT))
        self._weights = self._get_variable("weights",
                                          self.shape,
                                          self._get_initializer(self.init_para))
        if self.initial_bias_value is not None:
            self._biases = self._get_variable(
                'biases',
                [self.out_channel_num],
                 initializer=self._get_initializer(init_para={"name": "constant",
                                                              "value": self.initial_bias_value}))
        else:
            self._biases = None

    def _forward(self, x):
        self._data = A.nn.conv1d(x, self.weights, self.biases, self.stride, self.padding,
                                 name="fmap" if self.summarize_output else None)
        return self._data

Conv1D = Convolution1DLayer


class ColorfulConvLayer(ConvolutionLayer):
    def __init__(self,
                 c_W_initializer={"name": "uniform_unit_scaling", "factor": 1},
                 equivariant=False,
                 same_ksize=False,
                 verbose=False,
                 use_bn=True,
                 **kwargs):
        """
        Args:
            c_W_initializer: dict
                By default, uniform_unit_scaling_initializer is used. The rationale is
                to make the output variance (ideally should be norm) of color map and
                the feature map the same: the factor is set to 1 = 1/sqrt(3) *
                sqrt(COLOR_CHANNEL_NUM) --- COLOR_CHANNEL_NUM is 3. It is mostly for ad
                hoc experimental, which means this convolution layer should use default
                initializers.
            equivariant: bool
                Generate equivariant feature map if True, otherwise, use max inference.
            same_ksize: bool
                Use the same ksize with the filter for color filtering. If
                None, use 1 X 1 filter.
            use_bn: bool
                Use BN before addition. It is supposed to be used for testing,
                given it is harder to get the test case when BN is involved.
            verbose: bool
                If True, do histogram summary on both convolution map and color
                map.
        """
        super(ColorfulConvLayer, self).__init__(**kwargs)
        self.c_W_initializer = c_W_initializer
        self.verbose = verbose
        self.equivariant = equivariant
        self.same_ksize = same_ksize
        self.use_bn = use_bn

    def _pre_forward(self, input, *args, **kwargs):
        super(ConvolutionLayer, self)._pre_forward(*args, **kwargs)
        self.input_shape = input[0].get_shape().as_list()

    def _setup(self):
        super(ColorfulConvLayer, self)._setup()
        if self.same_ksize:
            shape = [self.ksize[0], self.ksize[1], 3, self.out_channel_num]
        else:
            shape = [1, 1, 3, self.out_channel_num]
        self.color_W, c_W_loss = self._variable_with_weight_decay(
            "color_weights", shape, self.c_W_initializer)
        if c_W_loss is not None:
            self._loss += c_W_loss

        if self.use_bn:
            self.color_bn = BatchNormalizationLayer(channel_num=self.out_channel_num, name="color_bn")
            self.shape_bn = BatchNormalizationLayer(channel_num=self.out_channel_num, name="shape_bn")
        else:
            self.color_bn = lambda x: x
            self.shape_bn = lambda x: x

    def _forward(self, X_in):
        F = X_in[0]
        C = X_in[1]
        F_out = super(ColorfulConvLayer, self)._forward(F)
        F_out = self.shape_bn(F_out)
        if self.verbose and not self.is_val:
            self._data_summary(F_out, sparsity_summary=False)

        C_out = A.nn.depthwise_conv2d(C, self.color_W, 1, self.padding)
        shape = C_out.get_shape().as_list()
        shape = [shape[0], shape[1], shape[2], 3, self.out_channel_num]
        C_out = A.reshape(C_out, shape)
        C_out = self.color_bn(C_out)
        if self.equivariant:
            F_out = A.expand_dims(F_out, axis=-2)
            out = F_out + C_out
            out = A.reshape(out, [shape[0], shape[1], shape[2], -1])

            if self.verbose and not self.is_val:
                self._data_summary(C_out, sparsity_summary=False)
                self._data_summary(A.abs(C_out)/(A.abs(F_out) + A.abs(C_out)), sparsity_summary=False)
        else:
            C_max = A.reduce_max(C_out, axis=3)

            if self.verbose and not self.is_val:
                self._data_summary(C_max)

            out = F_out + C_max

        self._data = out

        return self._data


class EquivariantProjectionLayer(SynapseLayer):
    def __init__(self, g_size, strides=1,  **kwargs):
        """
        NOTE: for this layer, the input should be g_size * in_channel_num, so
        is output.
        """
        super(EquivariantProjectionLayer, self).__init__(**kwargs)
        self.g_size = g_size
        self.strides = strides

    def _setup(self):
        self.weights, self._loss = self._variable_with_weight_decay(
            "projection_weights", [self.g_size, 1, 1, self.in_channel_num, self.out_channel_num], self.init_para)
        self.biases = None

    def _forward(self, X_in):
        shape = X_in.get_shape().as_list()
        new_shape = [shape[0], shape[1], shape[2], self.g_size, -1]
        X_in = A.reshape(X_in, new_shape)
        X_g = A.unstack(X_in, 3)
        X_out_g = []
        for i in range(self.g_size):
            X_out_g.append(
                A.nn.conv2d(X_g[i],
                             self.weights[i, ...],
                             strides=self.strides,
                             padding="SAME"))

        X_out = A.pack(X_out_g, axis=3)
        shape = X_out.get_shape().as_list()
        X_out = A.reshape(X_out, shape=[shape[0], shape[1], shape[2], -1])
        self._data = X_out

        return X_out


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
    NAME = "IP"

    def _forward(self, input):
        input = self._reshape(input)
        ip = A.nn.inner_product(input,
                                self.weights,
                                bias=self.biases,
                                name='fmap' if self.summarize_output else None)
        self._data = ip

        return self._data

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
        input_shape = A.get_shape(input)
        in_channel_num = input_shape[1]
        # Check the input shape, if it is not 2D tensor, reshape all remaining
        # dimensions into one.
        reshaped_input = input
        if len(input_shape) != 2:
            in_channel_num = 1
            for i in input_shape[1:]:
                in_channel_num *= i
            flattened = A.reshape(input, [input_shape[0], in_channel_num])
            reshaped_input = flattened

        # Hold another addition info about the shape of input feature maps.
        self.in_shape = input_shape[1:]

        return reshaped_input

    def _setup(self):
        self.shape = [self.in_channel_num, self.out_channel_num]
        self._weights = self._get_variable(
            'weights', self.shape, self._get_initializer(self.init_para))

        if self.initial_bias_value is not None:
            self._biases = self._get_variable(
                'biases',
                shape=[self.out_channel_num],
                initializer=self._get_initializer({"name": "constant", "value": self.initial_bias_value}))
        else:
            self._biases = None

    @deprecated(reason="Legacy code.")
    def _on_update_stub(self, K_prev):
        # Save the metric for this RKHS
        self.metric = K_prev
        # Compute the metric for the next layer
        if len(self.in_shape) == 3:
            shape = self.in_shape[:]
            shape.insert(0, self.out_channel_num)
            W = A.reshape(self.weights, shape)
        else:
            W = self.weights

        self.K = A.nn.nn_riemannic_metric(K_prev, W, self.biases)


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

    def _setup(self):
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
           not inspect.ismodule(x)]
