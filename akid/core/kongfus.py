"""
This module contains different genres of Kong Fu to a `Kid` to practice,
aka different training algorithms and policies to train a network.
"""
import abc
import inspect

import tensorflow as tf

from .common import TRAINING_DYNAMICS_COLLECTION, LEARNING_RATE_TAG
from . import common
from .blocks import ShadowableBlock


class LearningRateScheme(object):
    exp_decay = 1
    placeholder = 2


class KongFu(ShadowableBlock):
    """
    An top level abstract class to compute gradients given a loss.

    All concrete sub-class should save the computed gradients to `_data`, which
    is exposed by the property `data`. The data should be the return of
    `tf.train.Optimizer.compute_gradients`, which is a list of (gradient,
    variable) pairs. Further processing of the gradients could be done anyway
    you want, such as doing gradient average for data parallelism, or gradient
    clipping.

    Any concrete `KongFu` should implement `_get_optimizer` to provide a
    concrete optimizer.
    """
    def __init__(self,
                 lr_scheme={"name": LearningRateScheme.exp_decay,
                            "base_lr": 0.01,
                            "decay_rate": 0.95,
                            "num_batches_per_epoch": 468,
                            "decay_epoch_num": 1},
                 **kwargs):
        """
        Only exponential decay policy is supported now. Learning rate decays to
        `decay_rate` of current value after `decay_epoch_num` number of epochs
        have passed.
        Args:
            lr_scheme: dict
                 Learning rate scheme to use. Two types are supported for now:
                     1. `exp_decay`: exponential decay. 'base_lr',
                        'decay_rate' are required parameters. Either
                        'decay_steps' or 'decay_epoch_num' should be
                        present. If using 'decay_epoch_num', an additional
                        parameter 'num_batches_per_epoch' should be passed
                        in. It is used to convert epoch number to step number.
                     2. `placeholder`: a placeholder that is supposed to pass
                        in by feed dict, so arbitrary hard coded learning rate
                        decay could be used. To use this scheme, you should
                        update `lr_value` of `KongFu` manually, which is
                        normally done by assigning value in some callback
                        functions. Alternatively, if you are using `KongFu`
                        standalone, you need to feed a value to
                        `KongFu.learning_rate`.
                 See the default value for an example usage.
        """
        # Since normally we do not care what the name of an optimizer is, just
        # give it a default name.
        if "name" not in kwargs:
            kwargs["name"] = "opt"

        super(KongFu, self).__init__(**kwargs)
        self.lr_scheme = lr_scheme

    def _setup(self):
        if self.lr_scheme["name"] is LearningRateScheme.exp_decay:
            base_lr = float(self.lr_scheme["base_lr"])
            decay_rate = self.lr_scheme["decay_rate"]
            if "decay_steps" not in self.lr_scheme:
                decay_epoch_num = self.lr_scheme["decay_epoch_num"]
                num_batches_per_epoch \
                    = self.lr_scheme["num_batches_per_epoch"]
                decay_steps = num_batches_per_epoch * decay_epoch_num
            else:
                decay_steps = self.lr_scheme["decay_steps"]

            with tf.variable_scope(common.global_var_scope, reuse=True):
                step = tf.get_variable(common.GLOBAL_STEP)

            self.learning_rate = tf.train.exponential_decay(
                base_lr,
                step,
                decay_steps,
                decay_rate,
                staircase=True)
        elif self.lr_scheme["name"] is LearningRateScheme.placeholder:
            self.learning_rate = tf.placeholder(tf.float32,
                                           shape=[],
                                           name='lrn')
            self.lr_value = None
        else:
            raise Exception("Learning rate scheme is not supported. Please"
                            " specify one from `LearningRateScheme`.")

        self.opt = self._get_optimizer(self.learning_rate)

    def _forward(self, loss):
        """
        Build and return training ops according to the loss.
        """
        self._data = self.opt.compute_gradients(loss)

    def _post_setup(self):
        if self.do_summary:
            tf.summary.scalar(LEARNING_RATE_TAG,
                              self.learning_rate,
                              collections=[TRAINING_DYNAMICS_COLLECTION])

    @property
    def data(self):
        return self._data

    @abc.abstractmethod
    def _get_optimizer(self, lr):
        """
        An abstract method to get an optimizer. Sub-class of `KongFu` should
        implement this method to return a concrete optimizer for training.

        Args:
            lr: tensor variable returned by tf.train.exponential_decay
                The periodically decayed learning rate.
        """
        raise NotImplementedError('Each sub-kongfu needs to implement this'
                                  'method to provide an optimizer!')


class MomentumKongFu(KongFu):
    def __init__(self, momentum=0.9, use_nesterov=False, **kwargs):
        super(MomentumKongFu, self).__init__(**kwargs)
        self.momentum = momentum
        self.use_nesterov = use_nesterov

    def _get_optimizer(self, lr):
        return tf.train.MomentumOptimizer(lr,
                                          self.momentum,
                                          use_nesterov=self.use_nesterov)


class GradientDescentKongFu(KongFu):
    def _get_optimizer(self, lr):
        return tf.train.GradientDescentOptimizer(lr)


class AdamKongFu(KongFu):
    def _get_optimizer(self, lr):
        return tf.train.AdamOptimizer(lr)


__all__ = [name for name, x in locals().items() if
           not inspect.ismodule(x) and not inspect.isabstract(x)]
