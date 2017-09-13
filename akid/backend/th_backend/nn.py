from __future__ import division


import torch as th
from torch.nn import functional as F

from .computational_graph import get_shape


def l2_loss(var):
    return th.sum(var * var)


def l1_loss(var):
    return th.sum(th.abs(var))


def conv2d(input, filter, strides, padding, name=None):
    shape = get_shape(filter)
    H, W = shape[-2], shape[-1]
    padding = padding_str2tuple(padding, H, W)
    return F.conv2d(input, filter, stride=tuple(strides), padding=padding)


def bias_add(v, b):
    return v + b


def padding_str2tuple(padding, H, W):
    """
    Convert padding from string to tuple. Thea meaning of string is from
    tensorflow.
    """
    if padding == 'VALID':
        padding = 0
    elif padding == 'SAME':
        padding = (H//2, W//2)
    else:
        raise ValueError("{} padding is not supported. Should be VALID or SAME".format(padding))

    return padding
