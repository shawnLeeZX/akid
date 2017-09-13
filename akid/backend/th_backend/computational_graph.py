"""
PyTorch backend for akid.
"""
import numpy as np
import torch as th
from torch.autograd import Variable
from torch import nn

from ..common import *


def get_variable(name, shape=None,
                 initializer=None, trainable=True,
                 shared=True):
    """
    `name` is not used in torch.

    `shared`, and `trainable` are not used yet. They are legacy code from
    tensorflow, which may be useful when torch is used for distributed
    training.
    """
    if not callable(initializer):
        shape = None
        t = th.Tensor(initializer)
    else:
        t = th.Tensor(initializer(shape))

    return Variable(t)


def standardize_data_format(data, old_format):
    """
    Stardardize data to PyTorch format, which is channel first.

    Args:
        data: Tensor or numpy array.
            The input data.
        old_format: str
            A string describe the original format. For example, if converting
            from Tensorflow, it would be 'hwio' for parameter. See
            `SUPPORT_DATA_FORMAT` and `SUPPORT_PARA_FORMAT` for supported
            strings.
    """
    if old_format not in SUPPORT_PARA_FORMAT \
       and old_format not in SUPPORT_DATA_FORMAT:
        raise ValueError("The data format {} is not well specified.".format(old_format))

    if old_format in SUPPORT_PARA_FORMAT:
        out_format = 'oihw'
    else:
        out_format = 'nchw'

    if type(data) == np.ndarray:
        return np.einsum('{}->{}'.format(old_format, out_format), data)
    else:
        raise ValueError("Type {} is not supported.".format(type(data)))


def init():
    """
    Does nothing. Just to be compatible with tensorflow backend.
    """
    pass


def close():
    """
    The same as `init()`.
    """
    pass


def eval(t):
    """
    Convert torch tensor to numpy array.
    """
    if type(t) is Variable:
        v = t.data.numpy()
    else:
        v = t.numpy

    return v


def Tensor(t, require_grad=False):
    t = th.Tensor(t)
    if require_grad:
        t =  Variable(t)
    return t


def mul(a, b, name=None):
    return a * b


def get_shape(t):
    if type(t) is Variable:
        return list(t.data.shape)
    else:
        return list(t.shape)
