from __future__ import absolute_import
import tensorflow as tf
import numpy as np

# Data format
# #########################################################################
SUPPORT_PARA_FORMAT = ['oihw', 'hwio']
SUPPORT_DATA_FORMAT = ['nchw', 'nhwc']


# Parallel
# #######################################################################
TIMEOUT = 1



# Backends
# #########################################################################
TF = 'tensorflow'
TORCH = 'pytorch'
_backends = [TF, TORCH]


# Debug flags
# #########################################################################
_debug = False
def set_debug_flag(v):
    global _debug
    _debug = v


def get_debug_flag():
    return _debug


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
# A dict to count the number of blocks instantiated. It is used for automatic
# naming.
_block_count = {}


def inc_block_count(block_class):
    try:
        _block_count[block_class] += 1
    except KeyError:
        _block_count[block_class] = 1


def get_block_count(block_class):
    return _block_count[block_class]


def reset_block_count():
    global _block_count
    _block_count = {}


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

# Block modes.
# #########################################################################
# Enum types to provide mode a block could be in.
class Mode:
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def check_mode(mode):
    if mode != Mode.TRAIN \
        and mode != Mode.VAL \
        and mode != Mode.TEST:
        raise ValueError("Mode `{}` is not support.".format(mode))

# Named objects.
# #########################################################################


class NamedValue(object):
    pass


class NamedTensorTuple(tuple, NamedValue):
    """
    Object to pass a tuple of tensors around. It enables the tuple to be named,
    thus, be identifiable by name.

    To create a `NamedTensorTuple`::

        t1, t2 = ... # Two tensors
        NamedTensorTuple("any_name", t1, t2)

    Arbitrary number of tuple members are supported.
    """
    def __init__(self, name=None, *args, **kwargs):
        assert type(name) is str, "Named should be string."
        self.name = name

    def __new__(cls, name, *args, **kwargs):
        return super(NamedTensorTuple, cls).__new__(cls, *args, **kwargs)


class NamedNumericValue(NamedValue):
    pass


class NamedScalar(np.float, NamedNumericValue):
    """
    Object to pass a scalar. It enables the scalar to be named, thus, be
    identifiable by name.

    To create a `NamedScalar`::

        s = ... # A scalar
        NamedScalar("any_name", s)
    """
    def __init__(self, name=None, *args, **kwargs):
        self.name = name

    def __new__(cls, name, *args, **kwargs):
        return super(NamedScalar, cls).__new__(cls, *args, **kwargs)


class NamedNdarray(np.ndarray, NamedNumericValue):
    """
    Object to pass a named numpy array.

    To create a `NamedNdarray`::

        a = ... # A numpy array
        NamedNdarray("any_name", a)

    Refer to https://docs.scipy.org/doc/numpy/user/basics.subclassing.html for
    how to subclass numpy array.
    """
    def __new__(cls, name, array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        # It also triggers a call to NamedNdarray.__array_finalize__
        obj = np.asarray(array).view(cls)
        # set the new 'info' attribute to the value passed
        obj.name = name
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(NamedNdarray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. NamedNdarray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(NamedNdarray):
        #    obj is arr
        #    (type(obj) can be NamedNdarray)
        # From new-from-template - e.g namedarr[:3]
        #    type(obj) is NamedNdarray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # NamedNdarray.__new__ constructor, but also with
        # arr.view(NamedNdarray).
        self.name = getattr(obj, 'name', None)
        # We do not need to return anything
