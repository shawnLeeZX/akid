import tensorflow as tf
import numpy as np

# Data format
# #########################################################################
SUPPORT_PARA_FORMAT = ['oihw', 'hwio']
SUPPORT_DATA_FORMAT = ['nchw', 'nhwc']


# Backends
# #########################################################################
TF = 'tensorflow'
TORCH = 'pytorch'
_backends = [TF, TORCH]


def available_backends():
    return _backends


variable_scope = tf.variable_scope
get_variable_scope = tf.get_variable_scope


def get_scope_name():
    return get_variable_scope().name


# Clock (global step)
# # #######################################################################
# The name assigned to the current training step. It is used to create
# universal access to current training step. This is similar to clock in a
# computer, but in distributed system, such synchronized clock should not
# exist, but it could understand as physical time, meaning how long this
# kid has been trained or born.
_step = 0


def get_step():
    return _step


def step():
    """
    Move to next step
    """
    global _step
    _step += 1


def reset_step():
    global _step
    _step = 0


def set_step(step):
    global _step
    _step = step


_epoch = 0

def get_epoch():
    return _epoch


def inc_epoch():
    global _epoch
    _epoch += 1


def reset_epoch():
    global _epoch
    _epoch = 0

# Computation options
# #########################################################################
# TODO: this option does not work for tensorflow backend yet.
_use_cuda = True


def use_cuda(v=None):
    """
    If `v` is not None, set the value of flag use cuda, otherwise, return the
    flag's value.
    """
    global _use_cuda
    if v is not None:
        _use_cuda = v
    else:
        return _use_cuda

# Naming
# #######################################################################
def is_name_the_same(name_1, name_2):
    """
    Return if the name refer to the same named tensor. It seems a trivial
    string match. However, it can check whether two tensors on different devices
    refer to the same tensor, which has different suffixes.
    """
    name_1 = name_1.split(":")[0]
    name_2 = name_2.split(":")[0]
    return name_1 == name_2

# Data type
# #########################################################################
# TODO: data type is not enforced for now. It is possible even the data type is
# set to float32, the actual data being processed is not single precision number.
class DTYPE(object):
    FLOAT32 = 1
    FLOAT64 = 2

dtype = DTYPE.FLOAT32

def set_dtype(d):
    """
    Set data type by passing `DTYPE.FLOAT32`. See class `DTYPE` for supported
    data types.
    """
    global dtype
    dtype = d


def get_np_dtype():
    if dtype == DTYPE.FLOAT32:
        return np.float32
    elif dtype == DTYPE.FLOAT64:
        return np.float64
    else:
        raise ValueError("Date type not supported.")
