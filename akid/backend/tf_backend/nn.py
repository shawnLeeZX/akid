import tensorflow as tf


def depthwise_conv2d(input, filter, strides, padding, name=None):
    return tf.nn.depthwise_conv2d(input, filter, strides, padding, name)
