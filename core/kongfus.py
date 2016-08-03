"""
This module contains different genres of Kong Fu to a `Survivor` to practice,
aka different training algorithms and policies to train a network.
"""
import sys
import abc
import inspect

import tensorflow as tf

from .common import TRAINING_DYNAMICS_COLLECTION, LEARNING_RATE_TAG


class KongFu(object):
    """
    An top level abstract class to create and hold a training op for a
    `Survivor` to train a network. It uses first order stochastic gradient
    descent optimization methods.

    Any concrete `KongFu` should implement `_get_optimizer` to provide a
    concrete optimizer.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 base_lr=0.01,
                 decay_rate=0.95,
                 decay_epoch_num=1):
        """
        Only exponential decay policy is supported now. Learning rate decays to
        `decay_rate` of current value after `decay_epoch_num` number of epochs
        have passed.
        Args:
            decay_epoch_num: a float number
                Configure when to decay learning rate.
            Others are self-evident.
        """
        self.base_lr = float(base_lr)
        self.decay_rate = decay_rate
        self.decay_epoch_num = decay_epoch_num

    def setup(self, survivor):
        """
        Build and return training ops.

        Args:
            survivor: Survivor
                A KongFu needs to know who is using it to suit its taste. A
                survivor is passed to provide necessary information such as how
                many batches are there in a epoch of survivor's sensor, and
                also take away anything that should be held.
        """
        decay_steps = int(
            survivor.sensor.num_batches_per_epoch_train * self.decay_epoch_num)
        learning_rate = tf.train.exponential_decay(
            self.base_lr,
            survivor.global_step_tensor,
            decay_steps,          # Decay step.
            self.decay_rate,                # Decay rate.
            staircase=True)
        self.learning_rate = learning_rate
        tf.scalar_summary(LEARNING_RATE_TAG,
                          learning_rate,
                          collections=[TRAINING_DYNAMICS_COLLECTION])
        # Use simple momentum for the optimization.
        optimizer = self._get_optimizer(learning_rate)
        grads = optimizer.compute_gradients(survivor.brain.loss_graph)
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(
                    var.op.name + '/gradients',
                    grad,
                    collections=[TRAINING_DYNAMICS_COLLECTION])

        self.train_op = optimizer.apply_gradients(
            grads, global_step=survivor.global_step_tensor)

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
        sys.exit()


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
