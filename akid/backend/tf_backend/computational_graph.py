"""
Tensorflow backend for akid.

NOTE
# #########################################################################
To use tensorflow backend to execute/evaluate any graph built, run `init`
beforehand.
"""
from __future__ import absolute_import
import sys

from uuid import uuid4
import numpy as np
import tensorflow as tf

from .. import computational_graph as cg
from akid.utils import glog as log


DATA_FORMAT = "HWC"
# Build a global session to use
# TODO: think how to offer a way to close the session.
# A bug in tensorflow: Exception AttributeError: "'NoneType' object has no attribute 'TF_DeleteStatus'" in <bound method Session.__del__ of <tensorflow.python.client.session.Session object at 0x7f43105e9dd0>> ignored
# Should be fixed by now: https://github.com/tensorflow/tensorflow/issues/3388
sess = None
saver = None
initialized = False

float32 = tf.float32


def init(continue_from_chk_point=False, model_dir=None):
    global sess
    global saver
    global initialized

    if not initialized:
        if len(tf.global_variables()) != 0:
            saver = tf.train.Saver(tf.global_variables())

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        if continue_from_chk_point:
            log.info("Recovering net from checkpoint %s." % model_dir)
            restore(model_dir)
        else:
            init  = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
            sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        initialized = True


def close():
    if sess is not None:
        sess.close()

    global initialized
    initialized = False


def get_variable(name=None, shape=None,
                 initializer=None, trainable=True,
                 shared=True):
    """
    Refer to the same function in PyTorch backend for semantics.
    """
    # Generate a random name if the name is not given
    if not name:
        name = get_random_name()

    if not callable(initializer):
        shape = None

    if shared:
        if hasattr(initializer, "native"):
            # Use naive initializer instead of that of tensorflow
            init = initializer(shape)
            shape = None
        else:
            init = initializer

        return tf.get_variable(name, shape,
                               initializer=init, trainable=trainable)
    else:
        raise NotImplementedError("Normal Variable creation has not been implemented yet.")


def get_all_variables():
    return tf.global_variables()


def save(path):
    saver.save(sess,
               path + "/checkpoint",
               global_step=cg.get_step())


def restore(path):
    checkpoint = tf.train.get_checkpoint_state(path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        filename = checkpoint.model_checkpoint_path.split('/')[-1]
        step = int(filename.split('-')[-1])
        cg.set_step(step)
    else:
        log.error("No checkpoint found under %s!" % path)
        sys.exit()



def get_name(v):
    """
    Refer to the same function of PyTorch backend for docs.
    """
    if v is not None:
        return v.op.name.split(':')[0]
    else:
        return None


def remove_scope_from_name(name):
    """
    Remove scope and device information in the name.
    """
    name = name.split('/')[-1]

    return name


def append_suffix(name, suffix):
    """
    Append suffix to the name of a tensor, above the level of devices.
    """
    return name + '/' + suffix


def Tensor(X_in, requires_grad=False):
    """
    Get a tensor from a constant (array).

    Args:
        requires_grad: It is for compatibility with PyTorch. Not used here,
            since in tensorflow tensor can participate in auto-differentiation.
    """
    return tf.constant(X_in, dtype=tf.float32)


def eval(V):
    """
    Convert variable to numpy array.
    """
    if type(V) is np.ndarray:
        return V

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


def mean(v, name=None):
    return tf.reduce_mean(v, name=name)


def add_n(l, name=None):
    return tf.add_n(l, name=name)


def div(v, denominator, name=None):
    return tf.div(v, denominator, name=name)


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
    if old_format not in cg.SUPPORT_PARA_FORMAT \
       and old_format not in cg.SUPPORT_DATA_FORMAT:
        raise ValueError("The data format {} is not well specified.".format(old_format))

    if old_format in cg.SUPPORT_PARA_FORMAT:
        out_format = 'hwio'
    else:
        out_format = 'nhwc'

    if type(data) == np.ndarray:
        return np.einsum('{}->{}'.format(old_format, out_format), data)
    else:
        raise ValueError("Type {} is not supported.".format(type(data)))


def get_random_name():
    return uuid4().hex


def scatter(data, devices, name=None):
    # Since tensorflow handles all communications, just split the data with the
    # right number.
    num_split = len(devices)
    return tf.split(axis=0, num_or_size_splits=num_split, value=data, name=name)
