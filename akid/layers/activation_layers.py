import sys
import inspect

import tensorflow as tf
from tensorflow.python.training import moving_averages

from ..core.blocks import ProcessingLayer
from ..core import common
from .. import backend as A
import helper_methods


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
        self.ksize = helper_methods.expand_kernel(ksize)
        self.strides = helper_methods.expand_kernel(strides)
        self.padding = padding
        self.type = type

    def _forward(self, input):
        self.log("Padding method {}.".format(self.padding), debug=True)
        self.log("Pooling method {}.".format(self.type), debug=True)
        if self.type == "max":
            self._data = tf.nn.max_pool(input,
                                        self.ksize,
                                        self.strides,
                                        self.padding)
        elif self.type == "avg":
            self._data = tf.nn.avg_pool(input,
                                        self.ksize,
                                        self.strides,
                                        self.padding)
        else:
            raise Exception("Type `{}` pooling is not supported.".format(
                self.type))

        return self._data


class MaxPoolingLayer(ProcessingLayer):
    def __init__(self, ksize, strides, padding="VALID", get_argmax_idx=False, **kwargs):
        """
        Args:
        get_argmax_idx: Bool
                Compute the indices where the max has been obtained or not. The
                indices is saved in `self.in_group_indices`.
        """
        super(MaxPoolingLayer, self).__init__(**kwargs)

        self.ksize = helper_methods.expand_kernel(ksize)
        self.strides = helper_methods.expand_kernel(strides)

        self.padding = padding
        self.get_argmax_idx = get_argmax_idx

    def _forward(self, X_in):
        self.log("Padding method {}.".format(self.padding), debug=True)

        if type(X_in) is list:
            ret = []
            for X in X_in:
                ret.append(self._max_pooling(X))

            if self.get_argmax_idx:
                ret = zip(*ret)
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


class ReLULayer(ProcessingLayer):
    def _forward(self, input):
        self._data = tf.nn.relu(input)
        return self._data

    def backward(self, X_in):
        assert hasattr(self, "_data") and self._data is not None,\
            "Call forward first prior to backward"

        sign = A.cast(self._data > 10e-6, dtype=A.float32)  # Weed out very small values
        # NOTE: This can also implemented using backward propagated gradient of
        # ReLU. Try changing it when things go slow.
        self._data_g = X_in * sign

        return self._data_g


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
        for i in xrange(3):
            C_list.append(F * A.expand_dims(C[..., i], -1))
        out = A.concat(
            concat_dim=3,
            values=C_list)
        if self.wipe_negative:
            out = A.nn.relu(out)

        self._data = out

        return self._data


class SigmoidLayer(ProcessingLayer):
    def _forward(self, input):
        self._data = tf.nn.sigmoid(input)


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
        for i in xrange(0, len(data_split)):
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
    NAME = "Batch_Normalization"

    def __init__(self,
                 channel_num,
                 beta_init=0,
                 gamma_init=1,
                 fix_gamma=False,
                 share_gamma=False,
                 use_reference_bn=False,
                 **kwargs):
        super(BatchNormalizationLayer, self).__init__(**kwargs)
        self.channel_num = channel_num
        self.beta_init = float(beta_init)
        self.gamma_init = float(gamma_init)
        self.fix_gamma = fix_gamma
        self.share_gamma = share_gamma
        self.use_reference_bn = use_reference_bn

    def _setup(self):
        if self.use_reference_bn:
            self.log("Using reference BN. `beta_init` is fixed to 0;"
                     " `gamma_init` to 1.")
            self._data = self._ref_batch_norm(input)
            return

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

        # Bookkeeping a moving average for inference.

        # Since the initial mean and average are not accurate, we should use a
        # lower lower momentum. This is particularly important for ResNet since
        # the initial activation could be very large due to the exponential
        # accumulation effect of merge layers, though it does not work not well
        # to remove the effect for ResNet. To achieve this, we use the
        # mechanism provided by tensorflow, by passing current step in.
        with tf.variable_scope(common.global_var_scope, reuse=True):
            step = tf.get_variable(common.GLOBAL_STEP)
        self.ema = tf.train.ExponentialMovingAverage(0.9, step)


    def _forward(self, input):
        input_shape = input.get_shape().as_list()
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[-1]
        mean, variance = tf.nn.moments(input, reduction_axes)

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            # NOTE: Prior to tf 0.12, I did not need the variable scope above
            # this line to get things working. The problem is that moving
            # average cannot be put in a variable scope that plans to reuse its
            # variables (due to a debias mechanism introduced in tf 0.12). This
            # is a temporary fix, waiting for solutions in issues:
            # https://github.com/tensorflow/tensorflow/issues/6270 and
            # https://github.com/tensorflow/tensorflow/issues/5827
            ema_apply_op = self.ema.apply([mean, variance])
        ema_mean, ema_var = self.ema.average(mean), self.ema.average(variance)
        # Add the moving average to var list, for purposes such as
        # visualization.
        self.var_list.extend([ema_mean, ema_var])
        with tf.control_dependencies(
                [ema_apply_op]):
            if self.is_val:
                bn_input = self._bn(
                    input,
                    ema_mean,
                    ema_var,
                    self.beta,
                    self.gamma,
                    1e-5)
            else:
                bn_input = self._bn(
                    input,
                    mean,
                    variance,
                    self.beta,
                    self.gamma,
                    1e-5)

        self._data = bn_input

    def _bn(self, input, mean, variance, beta, gamma, epsilon):
        shape = input.get_shape().as_list()
        if len(shape) is 2 or self.share_gamma:
            normalized_input = (input - mean) / tf.sqrt(variance + epsilon)
            if self.gamma_init:
                normalized_input *= gamma
            bn_input = tf.add(normalized_input,
                              beta,
                              name=BatchNormalizationLayer.NAME)
        else:
            bn_input = tf.nn.batch_norm_with_global_normalization(
                input,
                mean,
                variance,
                beta,
                gamma,
                epsilon,
                True if self.gamma_init else False,
                name=BatchNormalizationLayer.NAME)

        return bn_input

    def backward(self, X_in):
        """
        Just pass the data back for now.
        """
        self._data = X_in

        return self._data

    def _ref_batch_norm(self, x):
        """
        Batch normalization from
        https://github.com/tensorflow/models/tree/master/resnet.

        It is introduced here for debugging purpose --- to see whether my
        implementation is wrong or not.
        """
        params_shape = [x.get_shape()[-1]]

        beta = self._get_variable(
            'beta',
            params_shape,
            initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable(
            'gamma',
            params_shape,
            initializer=tf.constant_initializer(1.0, tf.float32))

        if not self.is_val:
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

            moving_mean = self._get_variable(
                'moving_mean', params_shape,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)

            self._train_op = []
            self._train_op.append(moving_averages.assign_moving_average(
                moving_mean, mean, 0.9))
            self._train_op.append(moving_averages.assign_moving_average(
                moving_variance, variance, 0.9))
        else:
            mean = self._get_variable(
                'moving_mean',
                params_shape,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            variance = tf.get_variable(
                'moving_variance',
                params_shape,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)

        # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper
        # net.
        y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
        y.set_shape(x.get_shape())
        return y


class DropoutLayer(ProcessingLayer):
    def __init__(self, keep_prob=0.5, **kwargs):
        super(DropoutLayer, self).__init__(**kwargs)
        self.keep_prob = keep_prob

    def _forward(self, input):
        if self.is_val:
            self._data = tf.identity(input)
        else:
            self._data = tf.nn.dropout(input, self.keep_prob, seed=common.SEED)


__all__ = [name for name, x in locals().items() if not inspect.ismodule(x)]
