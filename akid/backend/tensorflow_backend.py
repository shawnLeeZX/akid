"""
Tensorflow backend for akid.

 Usage
================================================================================
To use tensorflow backend to execute/evaluate any graph built, run `init`
beforehand.
"""


import tensorflow as tf


# Build a global session to use
# TODO: think how to offer a way to close the session.
# A bug in tensorflow: Exception AttributeError: "'NoneType' object has no attribute 'TF_DeleteStatus'" in <bound method Session.__del__ of <tensorflow.python.client.session.Session object at 0x7f43105e9dd0>> ignored
# Should be fixed by now: https://github.com/tensorflow/tensorflow/issues/3388
sess = tf.Session()


def init():
    init  = tf.group(tf.global_variables_initializer(),
              tf.local_variables_initializer())
    sess.run(init)


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


def Tensor(X_in):
    """
    Get a tensor from a constant (array).
    """
    return tf.constant(X_in)


def eval(V):
    """
    Convert variable to numpy array.
    """
    return sess.run(V)
