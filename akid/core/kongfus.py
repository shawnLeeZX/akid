"""
This module contains different genres of Kong Fu to a `Kid` to practice,
aka different training algorithms and policies to train a network.
"""
from __future__ import absolute_import
import abc
import inspect

import tensorflow as tf

from .common import TRAINING_DYNAMICS_COLLECTION, LEARNING_RATE_TAG
from .blocks import ShadowableBlock
from .interface_blocks import UpdateBlock
from .. import backend as A


class LearningRateScheme(object):
    exp_decay = 1
    placeholder = 2

class KongFu(ShadowableBlock, UpdateBlock):
    """
    An top level abstract class to compute gradients given a loss.  Any
    concrete `KongFu` should implement `_get_optimizer` to provide a concrete
    optimizer.

    To change learning rate call `set_lr`. For example, to set a learning rate
    scheduler during training::

        kid = Kid(...)
        from akid import backend as A

        def update_lr(kid):
            if A.get_step() < 200:
                kid.kongfu.set_lr(0.1)
            elif A.get_step() < 400:
                kid.kongfu.set_lr(0.01)
            elif A.get_step() < 600:
                kid.kongfu.set_lr(0.001)
            else:
                kid.kongfu.set_lr(0.0001)

        kid.hooks.on_batch_begin.append(update_lr)

    The data of this block depends on backends.

    For PyTorch, since the gradients are saved along with variable
    tensors. `data` property is None.

    For Tensorflow: all concrete sub-class should save the computed gradients
    to `_data`, which is exposed by the property `data`. The data should be the
    return of `tf.train.Optimizer.compute_gradients`, which is a list of
    (gradient, variable) pairs. Further processing of the gradients could be
    done anyway you want, such as doing gradient average for data parallelism,
    or gradient clipping.
    """
    NAME = "Kongfu"

    def __init__(self,
                 lr=0.01,
                 var_list=None,
                 lr_scheme={"name": LearningRateScheme.exp_decay,
                            "base_lr": 0.01,
                            "decay_rate": 0.95,
                            "num_batches_per_epoch": 468,
                            "decay_epoch_num": 1},
                 **kwargs):
        """
        Args:
            var_list: list
                The list of variables that are supposed to train.
            lr: float
                The learning rate.
            lr_scheme: dict.
                This option is deprecated. Use lr instead. This option is
                supported with 'tensorflow' backend. To use it, `lr` needs to
                be set to None explicitly.

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
                       `KongFu._lr_tensor`.
                See the default value for an example usage.
        """
        super(KongFu, self).__init__(**kwargs)
        self._lr_value = lr
        if var_list is None:
            self.var_list = []
        else:
            self.var_list = var_list

        # Deprecated. Save for compatibility.
        if lr is None:
            self.lr_scheme = lr_scheme
        else:
            self.lr_scheme = None

    def _setup(self):
        if self.lr_scheme:
            # For compatibility.
            self._setup_lr_scheme()
        else:
            if A.backend() == A.TF:
                self._lr_tensor = tf.placeholder(tf.float32,
                                            shape=[],
                                            name='lrn')
                self.opt = self._get_optimizer(self._lr_tensor)
            elif A.backend() == A.TORCH:
                self.opt = self._get_optimizer(self._lr_value)
                A.cache_tensor(self._lr_value, LEARNING_RATE_TAG)

    def set_lr(self, lr):
        """
        Set the learning rate.
        """
        self._lr_value = lr

        if A.backend() == A.TORCH:
            if self.is_setup:
                for pg in self.opt.param_groups:
                    pg['lr'] = lr
            A.cache_tensor(self._lr_value, LEARNING_RATE_TAG)

    def set_var_list(self, var_list):
        self.var_list = var_list

    def append_var_list(self, var_list):
        self.var_list.extend(var_list)

    def get_lr(self):
        return self._lr_value

    def _setup_lr_scheme(self):
        if A.backend() != A.TF:
            raise ValueError("'lr_scheme' is supported only with tensorflow backend.")

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

            # TODO: step is not a tensor now, it may not be incremented during
            # optimization, which means the exponential decay will be broken ...
            step = A.get_step()

            self._lr_tensor = tf.train.exponential_decay(
                base_lr,
                step,
                decay_steps,
                decay_rate,
                staircase=True)
        elif self.lr_scheme["name"] is LearningRateScheme.placeholder:
            self._lr_tensor = tf.placeholder(tf.float32,
                                        shape=[],
                                        name='lrn')
            self._lr_value = None
        else:
            raise Exception("Learning rate scheme is not supported. Please"
                            " specify one from `LearningRateScheme`.")

    def _forward(self, loss):
        """
        Build and return training ops according to the loss.
        """
        self._data = A.train.compute_gradients(self.opt, loss)
        return self._data

    def _update(self, grads):
        return A.train.apply_gradients(self.opt, grads)

    def _post_forward(self, *args, **kwargs):
        if self.done_first_pass:
            return

        if self.do_summary:
            if A.backend() == A.TF:
                A.summary.scalar(LEARNING_RATE_TAG,
                                self._lr_tensor,
                                collections=[TRAINING_DYNAMICS_COLLECTION])
            elif A.backend() == A.TORCH:
                A.summary.scalar(LEARNING_RATE_TAG,
                                self._lr_value,
                                collections=[TRAINING_DYNAMICS_COLLECTION])

    def get_feed_dict(self):
        """
        Return a dict that fills the learning rate placeholder with current
        learning rate. Only useful when the backend is tensorflow.
        """
        return {self._lr_tensor: self._lr_value}

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
        return A.train.MomentumOptimizer(lr,
                                         self.var_list,
                                         momentum=self.momentum,
                                         use_nesterov=self.use_nesterov)


class GradientDescentKongFu(KongFu):
    def _get_optimizer(self, lr):
        return A.train.GradientDescentOptimizer(lr,
                                                self.var_list)


class AdamKongFu(KongFu):
    def _get_optimizer(self, lr):
        return A.train.AdamOptimizer(lr)


__all__ = [name for name, x in locals().items() if
           not inspect.ismodule(x) and not inspect.isabstract(x)]
