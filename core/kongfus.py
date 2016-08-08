"""
This module contains different genres of Kong Fu to a `Kid` to practice,
aka different training algorithms and policies to train a network.
"""
import abc
import inspect

import tensorflow as tf

from .common import TRAINING_DYNAMICS_COLLECTION, LEARNING_RATE_TAG
from .blocks import ShadowableBlock


class KongFu(ShadowableBlock):
    """
    An top level abstract class to create and hold a training op for a
    `Kid` to train a network. It uses first order stochastic gradient
    descent optimization methods.

    Any concrete `KongFu` should implement `_get_optimizer` to provide a
    concrete optimizer.
    """
    def __init__(self,
                 base_lr=0.01,
                 decay_rate=0.95,
                 decay_epoch_num=1,
                 **kwargs):
        """
        Only exponential decay policy is supported now. Learning rate decays to
        `decay_rate` of current value after `decay_epoch_num` number of epochs
        have passed.
        Args:
            decay_epoch_num: a float number
                Configure when to decay learning rate.
            Others are self-evident.
        """
        # Since normally we do not care what the name of an optimizer is, just
        # give it a default name.
        if "name" not in kwargs:
            kwargs["name"] = "opt"

        super(KongFu, self).__init__(**kwargs)
        self.base_lr = float(base_lr)
        self.decay_rate = decay_rate
        self.decay_epoch_num = decay_epoch_num

    def _setup(self, engine, loss):
        """
        Build and return training ops.

        Args:
            engine: Engine
                A KongFu needs to know who is using it to suit its taste. A
                engine is passed to provide necessary information such as how
                many batches are there in a epoch of engine's sensor, and
                also take away anything that should be held.
        """
        decay_steps = int(
            engine.sensor.num_batches_per_epoch_train * self.decay_epoch_num)
        learning_rate = tf.train.exponential_decay(
            self.base_lr,
            engine.global_step_tensor,
            decay_steps,          # Decay step.
            self.decay_rate,                # Decay rate.
            staircase=True)
        self.learning_rate = learning_rate
        self.opt = self._get_optimizer(learning_rate)
        self._data = self.opt.compute_gradients(loss)

    def _post_setup(self):
        if self.do_summary:
            tf.scalar_summary(LEARNING_RATE_TAG,
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
