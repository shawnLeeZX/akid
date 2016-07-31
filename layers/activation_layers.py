import sys
import inspect

import tensorflow as tf

from ..utils import glog as log
from ..core.blocks import ProcessingLayer
from ..core.common import SEED, GLOBAL_STEP, global_var_scope


class PoolingLayer(ProcessingLayer):
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

    def _setup(self, input):
        log.debug("Padding method {}.".format(self.padding))
        log.debug("Pooling method {}.".format(self.type))
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
            log.error("Type `{}` pooling is not supported.".format(
                self.type))


class ReLULayer(ProcessingLayer):
    def _setup(self, input):
        self._data = tf.nn.relu(input)


class SigmoidLayer(ProcessingLayer):
    def _setup(self, input):
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

    def _setup(self, input):
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

    def _setup(self, input):
        shape = input.get_shape().as_list()
        data = tf.reshape(input, [-1, shape[-1]])
        if self.use_temperature:
            T = self._get_variable("T",
                                   [1],
                                   initializer=tf.constant_initializer(10.0))
            data /= T
        if shape[-1] % self.group_size is not 0:
            log.error("Group size {} should evenly divide output channel"
                      " number {}".format(self.group_size, shape[-1]))
            sys.exit()
        num_split = shape[-1] // self.group_size
        log.info("Feature maps of layer {} is divided into {} group".format(
            self.name, num_split))
        data_split = tf.split(1, num_split, data)
        data_split = list(data_split)
        for i in xrange(0, len(data_split)):
            data_split[i] = tf.nn.softmax(data_split[i])
        data = tf.concat(1, data_split,)
        output = tf.reshape(data, shape, SoftmaxNormalizationLayer.NAME)

        self._data = output


class GroupSoftmaxLayer(ProcessingLayer):
    # A default name for the tensor returned by the layer.
    NAME = "GSMax"

    def __init__(self, use_temperature=False, group_size=4, **kwargs):
        super(GroupSoftmaxLayer, self).__init__(**kwargs)
        self.use_temperature = use_temperature
        self.group_size = group_size

    def _setup(self, input):
        shape = input.get_shape().as_list()
        if shape[-1] % self.group_size is not 0:
            log.error("Group size {} should evenly divide output channel"
                      " number {}".format(self.group_size, shape[-1]))
            sys.exit()
        out_channel_num = shape[-1]
        num_split = out_channel_num // self.group_size
        log.info("Feature maps of layer {} is divided into {} group".format(
            self.name, num_split))
        if num_split == out_channel_num:
            # Means the situation has degenerated into sigmoid activation
            output = tf.nn.sigmoid(input)
        else:
            data = tf.reshape(input, [-1, shape[-1]])
            if self.use_temperature:
                T = self._get_variable(
                    "T",
                    [1],
                    initializer=tf.constant_initializer(10.0))
                data /= T
            data_split = tf.split(1, num_split, data)
            data_split = list(data_split)
            # Augment each split with a constant 1.
            # Get dim of non-channel shape
            nc_shape = 1
            for d in shape[0:-1]:
                nc_shape *= d
            self.ground_state = tf.constant(1.0, shape=[nc_shape, 1])
            for i in xrange(0, len(data_split)):
                data_split[i] = tf.concat(1,
                                          [data_split[i],
                                           self.ground_state])
            for i in xrange(0, len(data_split)):
                data_split[i] = tf.nn.softmax(
                    data_split[i])[:, 0:self.group_size]
            data = tf.concat(1, data_split,)
            output = tf.reshape(data, shape, GroupSoftmaxLayer.NAME)

        self._data = output


class CollapseOutLayer(ProcessingLayer):
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

    def __init__(self, group_size=2, type="maxout", **kwargs):
        super(CollapseOutLayer, self).__init__(**kwargs)
        self.group_size = group_size
        self.type = type

    def _setup(self, input):
        shape = input.get_shape().as_list()
        if shape[-1] % self.group_size is not 0:
            log.error("Group size {} should evenly divide output channel"
                      " number {}".format(self.group_size, shape[3]))
            sys.exit()
        num_split = shape[-1] // self.group_size
        log.info("Feature maps of layer {} is divided into {} group".format(
            self.name, num_split))

        shape[-1] = num_split
        shape.append(self.group_size)
        buff_tensor = tf.reshape(input, shape)

        if self.type is "maxout":
            output = tf.reduce_max(buff_tensor,
                                   reduction_indices=len(shape)-1,
                                   name=CollapseOutLayer.MAXOUT_NAME)
        elif self.type is "average_out":
            output = tf.reduce_mean(buff_tensor,
                                    reduction_indices=len(shape)-1,
                                    name=CollapseOutLayer.AVEOUT_NAME)
        else:
            raise Exception("Type of `CollapseOutLayer` should be 'maxout' or"
                            "'average_out'! {} is given.".format(self.type))

        self._data = output


class BatchNormalizationLayer(ProcessingLayer):
    NAME = "Batch_Normalization"

    def __init__(self,
                 beta_init=0,
                 gamma_init=1,
                 fix_gamma=False,
                 share_gamma=False,
                 **kwargs):
        super(BatchNormalizationLayer, self).__init__(**kwargs)
        self.beta_init = float(beta_init)
        self.gamma_init = float(gamma_init)
        self.fix_gamma = fix_gamma
        self.share_gamma = share_gamma

    def _setup(self, input):
        # Logging.
        if self.gamma_init:
            log.info("Gamma initial value is {}.".format(self.gamma_init))
            if self.fix_gamma:
                log.info("Gamma is fixed to during training.")
            else:
                log.info("Gamma is trainable.")
        else:
            log.info("Gamma is not used during training.")

        input_shape = input.get_shape().as_list()
        if len(input_shape) is 2:
            mean, variance = tf.nn.moments(input, [0])
        else:
            mean, variance = tf.nn.moments(input, [0, 1, 2])
        beta = self._get_variable(
            'beta',
            shape=[input_shape[-1]],
            initializer=tf.constant_initializer(self.beta_init))
        if self.fix_gamma:
            gamma = tf.constant(
                self.gamma_init,
                shape=[] if self.share_gamma else [input_shape[-1]],
                name="gamma")
        else:
            gamma = self._get_variable(
                'gamma',
                shape=[] if self.share_gamma else [input_shape[-1]],
                initializer=tf.constant_initializer(self.gamma_init))

        # Bookkeeping a moving average for inference.

        # Since the initial mean and average are not accurate, we should use a
        # lower lower momentum. This is particularly important for ResNet since
        # the initial activation could be very large due to the exponential
        # accumulation effect of merge layers, though it does not work not well
        # to remove the effect for ResNet. To achieve this, we use the
        # mechanism provided by tensorflow, by passing current step in.
        with tf.variable_scope(global_var_scope, reuse=True):
            step = tf.get_variable(GLOBAL_STEP)
        ema = tf.train.ExponentialMovingAverage(0.99, step)

        ema_apply_op = ema.apply([mean, variance])
        ema_mean, ema_var = ema.average(mean), ema.average(variance)
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
                    beta,
                    gamma,
                    1e-5)
            else:
                bn_input = self._bn(
                    input,
                    mean,
                    variance,
                    beta,
                    gamma,
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


class DropoutLayer(ProcessingLayer):
    def __init__(self, keep_prob=0.5, **kwargs):
        super(DropoutLayer, self).__init__(**kwargs)
        self.keep_prob = keep_prob

    def _setup(self, input):
        if self.is_val:
            self._data = tf.identity(input)
        else:
            self._data = tf.nn.dropout(input, self.keep_prob, seed=SEED)


__all__ = [name for name, x in locals().items() if not inspect.ismodule(x)]
