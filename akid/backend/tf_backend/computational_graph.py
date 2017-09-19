"""
Tensorflow backend for akid.

NOTE
# #########################################################################
To use tensorflow backend to execute/evaluate any graph built, run `init`
beforehand.
"""
import numpy as np
import tensorflow as tf

from .. import computational_graph as cg_general


# Build a global session to use
# TODO: think how to offer a way to close the session.
# A bug in tensorflow: Exception AttributeError: "'NoneType' object has no attribute 'TF_DeleteStatus'" in <bound method Session.__del__ of <tensorflow.python.client.session.Session object at 0x7f43105e9dd0>> ignored
# Should be fixed by now: https://github.com/tensorflow/tensorflow/issues/3388
sess = None

float32 = tf.float32


def init():
    global sess
    sess = tf.Session()
    init  = tf.group(tf.global_variables_initializer(),
              tf.local_variables_initializer())
    sess.run(init)


def close():
    sess.close()
    tf.reset_default_graph()


def get_variable(name, shape=None,
                 initializer=None, trainable=True,
                 shared=True):
    if not callable(initializer):
        shape = None

    if shared:
        return tf.get_variable(name, shape,
                               initializer=initializer, trainable=trainable)
    else:
        raise NotImplementedError("Normal Variable creation has not been implemented yet.")


def get_name(v):
    """
    Only return the last part in the name hierarchy. For the reason, see the
    same function of PyTorch backend.
    """
    parts = v.op.name.split('/')
    name = parts[-1]
    return name


def Tensor(X_in, require_grad=False):
    """
    Get a tensor from a constant (array).

    Args:
        require_grad: It is for compatibility with PyTorch. Not used here,
            since in tensorflow tensor can participate in auto-differentiation.
    """
    return tf.constant(X_in, dtype=tf.float32)


def eval(V):
    """
    Convert variable to numpy array.
    """
    return sess.run(V)


def run(op, feed_dict=None):
    """Run an operation with arguments. For compatibility with Tensorflow"""
    return sess.run(op, feed_dict=feed_dict)


def split(split_dim, num_split, value, name="split"):
    return tf.split(axis=split_dim, num_or_size_splits=num_split, value=value, name="split")


def reshape(tensor, shape, name=None):
    return tf.reshape(tensor, shape, name)


def reduce_sum(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None):
    tf.reduce_sum(input_tensor, axis, keep_dims, name, reduction_indices)


def cast(x, dtype, name=None):
    return tf.cast(x, dtype, name)


def concat(concat_dim, values, name="concat"):
    return tf.concat(axis=concat_dim, values=values, name=name)


def stack(values, axis=0, name="pack"):
    return tf.stack(values, axis=axis, name=name)


def unstack(values, axis=0, name="unstack"):
    return tf.unstack(values, num=None, axis=axis, name=name)


def pack(values, axis=0, name="pack"):
    return tf.stack(values, axis=axis, name=name)


def expand_dims(input, axis=None, name=None):
    return tf.expand_dims(input, axis=axis, name=name)


def reduce_max(input_tensor, axis=None, keep_dims=False, name=None):
    return tf.reduce_max(input_tensor, axis=axis, keep_dims=keep_dims, name=name)


def abs(x, name=None):
    return tf.abs(x, name=name)


def is_tensor(T):
    return type(T) is tf.Tensor


def mul(a, b, name=None):
    return tf.multiply(a, b, name)


def add_n(l, name=None):
    return tf.add_n(l, name=name)


def zero_fraction(data):
    return tf.nn.zero_fraction(data)


def get_shape(t):
    return t.get_shape().as_list()


def standardize_data_format(data, old_format):
    """
    Stardardize data to Tensorflow format, which is first last.

    Args:
        data: Tensor or numpy array.
            The input data.
        old_format: str
            A string describe the original format. For example, if converting
            from Tensorflow, it would be 'hwio' for parameter. See
            `SUPPORT_DATA_FORMAT` and `SUPPORT_PARA_FORMAT` for supported
            strings.
    """
    if old_format not in cg_general.SUPPORT_PARA_FORMAT \
       and old_format not in cg_general.SUPPORT_DATA_FORMAT:
        raise ValueError("The data format {} is not well specified.".format(old_format))

    if old_format in cg_general.SUPPORT_PARA_FORMAT:
        out_format = 'hwio'
    else:
        out_format = 'nhwc'

    if type(data) == np.ndarray:
        return np.einsum('{}->{}'.format(old_format, out_format), data)
    else:
        raise ValueError("Type {} is not supported.".format(type(data)))
