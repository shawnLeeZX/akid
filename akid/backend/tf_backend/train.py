from __future__ import absolute_import
import tensorflow as tf


def compute_gradients(opt, v):
    return opt.compute_gradients(v)


def apply_gradients(opt, grad):
    return opt.apply_gradients(grad)


def MomentumOptimizer(lr, var_list=None, momentum=0.9, use_nesterov=False):
    return tf.train.MomentumOptimizer(lr, momentum, use_nesterov=use_nesterov)


def GradientDescentOptimizer(lr, var_list=None):
    return tf.train.GradientDescentOptimizer(lr)


def AdamOptimizer(lr, var_list=None):
    return tf.train.AdamOptimizer(lr)
