from __future__ import absolute_import
import sys
import inspect

import tensorflow as tf
from tensorflow.python.training import moving_averages

from torch.nn import functional as F

from ..core.blocks import ProcessingLayer
from .. import backend as A
from six.moves import range
from six.moves import zip


class PoolingLayer(ProcessingLayer):
    """
    Pooling layer that supports a type to choose what pooling methods to use.

    It is deprecated. Use specialized layer, e.g. `MaxpoolingLayer` instead.
    """
    def __init__(self, ksize, strides, padding="VALID", type="max", **kwargs):
        """
        Args:
            type: str
                Use max or average pooling. 'max' for max pooling, and 'avg'
                for average pooling.
        """
        super(PoolingLayer, self).__init__(**kwargs)
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.type = type

    def _setup(self):
        self.log("Padding method {}.".format(self.padding), debug=True)
        self.log("Pooling method {}.".format(self.type), debug=True)

    def _forward(self, input):
        if self.type == "max":
            self._data = A.nn.max_pool(input,
                                       self.ksize,
                                       self.strides,
                                       self.padding)
        elif self.type == "avg":
            self._data = A.nn.avg_pool(input,
                                       self.ksize,
                                       self.strides,
                                       self.padding)
        else:
            raise Exception("Type `{}` pooling is not supported.".format(
                self.type))

        return self._data


class MemoryMaxPoolingLayer(ProcessingLayer):
    NAME = "MPool"

    def __init__(self, ksize, strides, padding="VALID", type="max", **kwargs):
        """
        Args:
            type: str
                Use max or average pooling. 'max' for max pooling, and 'avg'
                for average pooling.
        """
        super(MemoryMaxPoolingLayer, self).__init__(**kwargs)
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.keep_memory = False
        self.memory = None

    def _setup(self):
        self.log("Padding method {}.".format(self.padding), debug=True)

    def _forward(self, input):
        if self.keep_memory:
            if self.memory is None:
                self._data, self.memory = A.nn.max_pool(input,
                                                        self.ksize,
                                                        self.strides,
                                                        self.padding,
                                                        return_indices=True)
            else:
                shape = A.get_shape(input)
                mshape = A.get_shape(self.memory)
                if mshape[:-2] != shape[:-2]:
                    if mshape[1:-2] != shape[1:-2]:
                        # Only the first dimension is allowed to be different.
                        raise ValueError("Shape not supported.")

                    idxs = A.cat([self.memory] * shape[0])
                else:
                    idxs = self.memory

                tshape = shape[:-2]
                tshape.append(shape[-1] * shape[-2])
                d = A.reshape(input, tshape)

                mshape = A.get_shape(idxs)
                mtshape = mshape[:-2]
                mtshape.append(mshape[-1] * mshape[-2])
                m = A.reshape(idxs, mtshape)

                out = A.gather(d, dim=-1, index=m)
                self._data = A.reshape(out, mshape)

        else:
            self._data, self.memory = A.nn.max_pool(input,
                                                    self.ksize,
                                                    self.strides,
                                                    self.padding,
                                                    return_indices=True)


        return self._data

    def set_state(self, state):
        self.keep_memory = state

MPoolLayer = MemoryMaxPoolingLayer


class _PoolingLayer(ProcessingLayer):
    def __init__(self, ksize, strides, padding, **kwargs):
        super(_PoolingLayer, self).__init__(**kwargs)

        self.ksize = ksize
        self.strides = strides
        self.padding = padding


class MaxPoolingLayer(_PoolingLayer):
    NAME = "MaxPool2D"

    def __init__(self, get_argmax_idx=False, **kwargs):
        """
        Args:
        get_argmax_idx: Bool
                Compute the indices where the max has been obtained or not. The
                indices is saved in `self.in_group_indices`.
        """
        super(MaxPoolingLayer, self).__init__(**kwargs)

        ksize = self.ksize
        t = type(ksize)
        if t is int:
            self.ksize = [ksize, ksize]
        elif t is list or t is tuple:
            if len(ksize) == 2:
                self.ksize = ksize
            else:
                raise ValueError("Only 2D pooling is supported. Gotten {}".format(ksize))
        else:
            raise ValueError("Value not understood".format(ksize))


        self.get_argmax_idx = get_argmax_idx

    def _forward(self, X_in):
        self.log("Padding method {}.".format(self.padding), debug=True)

        if type(X_in) is list:
            ret = []
            for X in X_in:
                ret.append(self._max_pooling(X))

            if self.get_argmax_idx:
                ret = list(zip(*ret))
                self._data, self.in_group_indices = ret[0], ret[1]
            else:
                self._data = ret
        else:
            ret = self._max_pooling(X_in)
            if self.get_argmax_idx:
                self._data, self.in_group_indices = ret
            else:
                self._data = ret

        return self._data

    def _max_pooling(self, X_in):
        if self.get_argmax_idx:
            return A.nn.max_pool_with_argmax(
                X_in, self.ksize, self.strides, self.padding)
        else:
            return A.nn.max_pool(X_in,
                                 self.ksize,
                                 self.strides,
                                 self.padding)



    def backward(self, X_in):
        if not self.get_argmax_idx:
            raise Exception("Set `get_argmax_idx` to True to use backward.")

        self._data_g = A.nn.max_unpooling(X_in, self.in_group_indices, self.ksize)

        return self._data_g


class MaxPooling1DLayer(_PoolingLayer):
    NAME = "MaxPooling1D"

    def _forward(self, x):
        self._data = A.nn.max_pool1d(x, self.ksize, self.strides, self.padding)
        return self._data

MaxPooling1D = MaxPooling1DLayer

class ReLULayer(ProcessingLayer):
    NAME = "ReLU"
    def _forward(self, input):
        self._data = A.nn.relu(
            input,
            name='fmap' if self.summarize_output else None)
        return self._data

    def backward(self, X_in):
        assert hasattr(self, "_data") and self._data is not None,\
            "Call forward first prior to backward"

        sign = A.cast(self._data > 10e-6, dtype=A.float32)  # Weed out very small values
        # NOTE: This can also implemented using backward propagated gradient of
        # ReLU. Try changing it when things go slow.
        self._data_g = X_in * sign

        return self._data_g

ReLU = ReLULayer


class MemoryReLULayer(ProcessingLayer):
    """
    ReLU layer that is able to memorize the activation patterns. If
    `keep_memory` is set to True, the layer will memorize the activation if it
    does not have one yet. The memorized activation pattern will be used until
    `keep_memory` is set to False.
    """

    NAME = "MReLU"
    def __init__(self, keep_memory=False, *args, **kwargs):
        super(MemoryReLULayer, self).__init__(*args, **kwargs)

        self.keep_memory = keep_memory
        self.memory = None

    def _forward(self, input):
        if self.keep_memory:
            if self.memory is None:
                self.memory = A.cast(input > 0, A.float32)
        else:
            self.memory = A.cast(input > 0, A.float32)

        self._data = A.mul(self.memory, input, name='fmap' if self.summarize_output else None)

        return self._data

    def set_state(self, keep_memory):
        self.keep_memory = keep_memory


MReLU = MemoryReLULayer


class ColorizationReLULayer(ProcessingLayer):
    def __init__(self, wipe_negative=False, **kwargs):
        super(ColorizationReLULayer, self).__init__(**kwargs)
        self.wipe_negative = wipe_negative

    def _forward(self, X_in):
        """
        This layer takes two inputs, the convoluted feature map and the color
        map that is downsampled accordingly to match the shape of the feature
        map.
        """
        F, C = X_in[0], X_in[1]
        F = A.nn.relu(F)
        C_list = []
        for i in range(3):
            C_list.append(F * A.expand_dims(C[..., i], -1))
        out = A.concat(
            concat_dim=3,
            values=C_list)
        if self.wipe_negative:
            out = A.nn.relu(out)

        self._data = out

        return self._data


class SigmoidLayer(ProcessingLayer):
    NAME = "Sigmoid"

    def _forward(self, x):
        self._data = A.nn.sigmoid(x)
        return self._data

Sigmoid = SigmoidLayer


class LRNLayer(ProcessingLayer):
    def __init__(self,
                 depth_radius=4,
                 bias=1,
                 alpha=0.001 / 9.0,
                 beta=0.75,
                 **kwargs):
        super(LRNLayer, self).__init__(**kwargs)
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def _forward(self, input):
        self._data = tf.nn.lrn(input,
                               self.depth_radius,
                               self.bias,
                               self.alpha,
                               self.beta)


class SoftmaxNormalizationLayer(ProcessingLayer):
    # A default name for the tensor returned by the layer.
    NAME = "Softmax_Normalization"

    def __init__(self, use_temperature=False, group_size=4, **kwargs):
        super(SoftmaxNormalizationLayer, self).__init__(**kwargs)
        self.use_temperature = use_temperature
        self.group_size = group_size

    def _forward(self, input):
        shape = input.get_shape().as_list()
        data = tf.reshape(input, [-1, shape[-1]])
        if self.use_temperature:
            T = self._get_variable("T",
                                   [1],
                                   initializer=tf.constant_initializer(10.0))
            data /= T
        if shape[-1] % self.group_size is not 0:
            raise Exception("Group size {} should evenly divide output channel"
                            " number {}".format(self.group_size, shape[-1]))
        num_split = shape[-1] // self.group_size
        self.log("Feature maps of layer {} is divided into {} group".format(
            self.name, num_split))
        data_split = tf.split(axis=1, num_or_size_splits=num_split, value=data)
        data_split = list(data_split)
        for i in range(0, len(data_split)):
            data_split[i] = tf.nn.softmax(data_split[i])
        data = tf.concat(axis=1, values=data_split,)
        output = tf.reshape(data, shape, SoftmaxNormalizationLayer.NAME)

        self._data = output


class GroupProcessingLayer(ProcessingLayer):
    """
    A abstract layer that processes layer by group. This is a meta class
    (`_forward` is not implemented).

    Two modes are possible for this layer. The first is to divide the
    neurons of this layer evenly using `group_size`. The second mode the
    input should be a list of tensors. In this case, `group_size` is
    ignored, and each tensor in the list is taken as a group. In both case,
    only the last dimension of the tensor indexes group member, the other
    dimensions index groups.
    """
    def __init__(self, group_size=4, **kwargs):
        super(GroupProcessingLayer, self).__init__(**kwargs)
        self.group_size = group_size
        # Members to be filled during `_pre_setup`.
        self.output_shape = None
        self.num_group = None
        self.shape_rank = None

    def _pre_forward(self, input):
        super(GroupProcessingLayer, self)._pre_forward(input)

        if type(input) is list:
            # Get the shape for the final output tensor.
            last_dim = 0
            for t in input:
                shape = t.get_shape().as_list()
                last_dim += shape[-1]
            self.output_shape = input[0].get_shape().as_list()
            self.output_shape[-1] = last_dim
            self.rank = len(self.output_shape)
            # No work actually done. Just logging and gather some meta data.
            self.num_group = len(input)
            self.log("Number of groups: {}".format(self.num_group))
            group_size_list = [t.get_shape().as_list()[-1] for t in input]
            self.log("Group size of each group are {}".format(group_size_list))
        else:
            self.output_shape = input.get_shape().as_list()
            self.rank = len(self.output_shape)
            if self.output_shape[-1] % self.group_size is not 0:
                raise Exception("Group size {} should evenly divide output channel"
                                " number {}".format(self.group_size,
                                                    self.output_shape[-1]))
            out_channel_num = self.output_shape[-1]
            self.num_group = out_channel_num // self.group_size
            self.log("Feature maps of layer {} is divided into {} group".format(
                self.name, self.num_group))
            self.log("All groups have equal size {}.".format(self.group_size))


class GroupSoftmaxLayer(GroupProcessingLayer):
    # A default name for the tensor returned by the layer.
    NAME = "GSMax"

    def __init__(self, concat_output=True, use_temperature=False, **kwargs):
        """
        Args:
            concat_output: Boolean
                Whether to concat the scattered list into one tensor.
        """
        super(GroupSoftmaxLayer, self).__init__(**kwargs)
        self.use_temperature = use_temperature
        self.concat_output = concat_output

    def _forward(self, input):
        # Divide the input into list if not already.
        if type(input) is list:
            splitted_input = input
        else:
            out_channel_num = self.output_shape[-1]
            if self.num_group == out_channel_num:
                # Means the situation has degenerated into sigmoid activation
                # Just compute and return
                self._data = tf.nn.sigmoid(input)
                return

            splitted_input = tf.split(axis=self.rank-1, num_or_size_splits=self.num_group, value=input)
            splitted_input = list(splitted_input)

        # Add temperature if needed
        if self.use_temperature:
            for t in splitted_input:
                T = self._get_variable(
                    "T",
                    [1],
                    initializer=tf.constant_initializer(10.0))
                t /= T

        for i, t in enumerate(splitted_input):
            # Augment each split with a constant 1.
            ground_state_shape = self.output_shape[0:-1]
            ground_state_shape.append(1)
            self.ground_state = tf.constant(1.0, shape=ground_state_shape)

            # Compute group softmax.
            augmented_t = tf.concat(axis=self.rank-1, values=[t, self.ground_state])
            softmax_t = tf.nn.softmax(augmented_t)
            splitted_input[i] = softmax_t[..., 0:-1]

        output = splitted_input

        if self.concat_output:
            output = tf.concat(axis=self.rank-1, values=splitted_input)

        self._data = output


class CollapseOutLayer(GroupProcessingLayer):
    """
    `CollapseOutLayer` is to collapse the a subspace into a one-dimensional
    space. It could be Maxout or AverageOut. To ensure backward compatibility,
    by default, Maxout is the choice.

    It is not merged into PoolingLayer is because CollapseOutLayer should
    strictly use `VALID` padding so to decouple these two type of padding,
    these two layers are separated.
    """
    # A default name for the tensor returned by the layer.
    MAXOUT_NAME = "MaxOut"
    AVEOUT_NAME = "AverageOut"

    def __init__(self, type="maxout", **kwargs):
        super(CollapseOutLayer, self).__init__(**kwargs)
        self.type = type

    def _forward(self, input):
        if type(input) is list:
            # process each tensor once by one and combine them
            reduced_t_list = []
            for t in input:
                reduced_t = self._reduce(t)
                reduced_t_list.append(reduced_t)

            output = tf.stack(reduced_t_list, axis=-1)
        else:
            shape_by_group = self.output_shape[:]
            shape_by_group[-1] = self.num_group
            shape_by_group.append(self.group_size)
            buff_tensor = tf.reshape(input, shape_by_group)
            output = self._reduce(buff_tensor)

        self._data = output

    def _reduce(self, tensor):
        shape = tensor.get_shape().as_list()
        if self.type is "maxout":
            output = tf.reduce_max(tensor,
                                   axis=len(shape)-1,
                                   name=CollapseOutLayer.MAXOUT_NAME)
        elif self.type is "average_out":
            output = tf.reduce_mean(tensor,
                                    axis=len(shape)-1,
                                    name=CollapseOutLayer.AVEOUT_NAME)
        else:
            raise Exception("Type of `CollapseOutLayer` should be 'maxout' or"
                            "'average_out'! {} is given.".format(self.type))

        return output


class BatchNormalizationLayer(ProcessingLayer):
    NAME = "BN"

    def __init__(self,
                 channel_num,
                 beta_init=0,
                 gamma_init=1,
                 eps=1e-5,
                 momentum=0.1,
                 track_running_stats=True,
                 fix_gamma=False,
                 share_gamma=False,
                 dim_num=None,
                 **kwargs):
        """
        `beta_init`, `gamma_init`, `share_gamma` are only useful in tensorflow
        backend.
        """
        super(BatchNormalizationLayer, self).__init__(**kwargs)
        self.channel_num = channel_num
        self.fix_gamma = fix_gamma
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        # TODO: make the init setter work for torch as well.
        if A.backend() == A.TF:
            self.beta_init = float(beta_init)
            self.gamma_init = float(gamma_init)
            self.share_gamma = share_gamma

    def _setup(self):
        if A.backend() == A.TORCH:
            self.weights = self._get_variable("weights",
                                            self.channel_num,
                                            self._get_initializer({"name": "range_uniform", "low": 0, "high": 1}))
            self.biases = self._get_variable("biases",
                                           self.channel_num,
                                           self._get_initializer({"name": "constant", "value": 0}))

            if self.track_running_stats:
                # Running mean and variance that require no grad.
                self.running_mean = self._get_variable("running_mean",
                                                    self.channel_num,
                                                    self._get_initializer({"name": "constant", "value": 0}),
                                                    trainable=False)
                self.running_var = self._get_variable("running_var",
                                                    self.channel_num,
                                                    self._get_initializer({"name": "constant", "value": 1}),
                                                    trainable=False)
                self.num_batches_tracked = self._get_variable("num_batches_tracked",
                                                            0,
                                                            self._get_initializer({"name": "scalar", "value": 0}),
                                                            trainable=False)
            else:
                self.running_mean = None
                self.running_var = None
                self.num_batches_tracked = None

            return

        elif A.backend() == A.TF:
            # Logging.
            if self.gamma_init:
                self.log("Gamma initial value is {}.".format(self.gamma_init))
                if self.fix_gamma:
                    self.log("Gamma is fixed to during training.")
                else:
                    self.log("Gamma is trainable.")
            else:
                self.log("Gamma is not used during training.")

            self.beta, loss = self._variable_with_weight_decay(
                'beta',
                shape=[self.channel_num],
                init_para={"name": "constant", "value": self.beta_init})

            if loss is not None:
                self._loss = loss

            if self.fix_gamma:
                self.gamma = tf.constant(
                    self.gamma_init,
                    shape=[] if self.share_gamma else [self.channel_num],
                    name="gamma")
            else:
                self.gamma, loss = self._variable_with_weight_decay(
                    'gamma',
                    shape=[] if self.share_gamma else [self.channel_num],
                    init_para={"name": "constant", "value": self.gamma_init})

                if loss is not None:
                    self._loss += loss
        else:
            raise ValueError("Backend {} is not supported".format(A.backend()))


    def _forward(self, input):
        # TODO: the backend dependent code should be changed to a backend
        # independent one.
        if A.backend() == A.TF:
            self._data = A.nn.bn(input,
                                self.channel_num,
                                self.gamma,
                                self.beta,
                                self.is_val,
                                A.get_step(),
                                fix_gamma=self.fix_gamma,
                                share_gamma=self.share_gamma,
                                name="bn" if self.summarize_output else None)
        elif A.backend() == A.TORCH:
            exponential_average_factor = 0.0

            if not self.is_val and self.track_running_stats:
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            self._data = A.nn.bn(input, self.channel_num,
                                 self.weights, self.biases,
                                 self.running_mean, self.running_var,
                                 self.is_val, self.track_running_stats,
                                 exponential_average_factor, self.eps,
                                 name="bn" if self.summarize_output else None)

        return self._data

    def backward(self, X_in):
        """
        Just pass the data back for now.
        """
        self._data = X_in

        return self._data

BN = BatchNormalizationLayer


class DropoutLayer(ProcessingLayer):
    NAME = "Dropout"

    def __init__(self, keep_prob=0.5, **kwargs):
        super(DropoutLayer, self).__init__(**kwargs)
        self.keep_prob = keep_prob

    def _forward(self, input):
        self._data = A.nn.dropout(input, self.keep_prob, self.is_val)

Dropout = DropoutLayer


class FixedWeightInnerProductLayer(ProcessingLayer):
    NAME = "FixedWeightIP"

    def __init__(self, weights, **kwargs):
        super(FixedWeightInnerProductLayer, self).__init__(**kwargs)
        self.weights = weights

    def _setup(self):
        self.weights = A.Tensor(self.weights)

    def _forward(self, x):
        self._data = A.nn.inner_product(x,
                                        self.weights,
                                        name='fmap' if self.summarize_output else None)

        return self._data

FixedWeightIPLayer = FixedWeightInnerProductLayer


class L2NormalizationLayer(ProcessingLayer):
    NAME = "L2Norm"

    def _forward(self, x):
        dim = len(x.shape)
        assert dim == 2, "dim {} is not supported".format(dim)

        norm = A.nn.l2_norm(x, 1)
        norm = A.expand_dims(norm, axis=1)
        self._data = x / norm

        return self._data

L2Norm = L2NormalizationLayer


__all__ = [name for name, x in locals().items() if not inspect.ismodule(x)]
