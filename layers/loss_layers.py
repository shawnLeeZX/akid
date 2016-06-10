import inspect

import tensorflow as tf

from ..core.blocks import ProcessingLayer

NUM_CLASSES = 10


class LossLayer(ProcessingLayer):
    """
    An abstract top level loss layer.

    It is nothing here now. However, it has already been used to check whether
    a layer is a loss layer or not.
    """
    pass


class SoftmaxWithLossLayer(LossLayer):
    def _setup(self, logits, labels):
        # Convert from sparse integer labels in the range [0, NUM_CLASSSES) to
        # 1-hot dense float vectors (that is we will have batch_size vectors,
        # each with NUM_CLASSES values, all of which are 0.0 except there will
        # be a 1.0 in the entry corresponding to the label).
        batch_size = tf.size(labels)
        labels = tf.expand_dims(labels, 1)
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        concated = tf.concat(1, [indices, labels])
        onehot_labels = tf.sparse_to_dense(
            concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
        cross_entropy \
            = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                      onehot_labels,
                                                      name='xentropy')
        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='xentropy_mean')
        self._loss = cross_entropy_mean

__all__ = [name for name, x in locals().items() if
           not inspect.ismodule(x) and not inspect.isabstract(x)]
