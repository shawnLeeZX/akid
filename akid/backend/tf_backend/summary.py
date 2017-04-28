import tensorflow as tf


def histogram(name, values, collections=None):
    tf.summary.histogram(name, values, collections)
