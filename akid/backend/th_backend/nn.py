from __future__ import division


import torch as th
from torch.nn import functional as F

from .computational_graph import cache_name_if_exist
from . import computational_graph as cg


@cache_name_if_exist
def l2_loss(var, name=None):
    return th.sum(var * var)


@cache_name_if_exist
def l1_loss(var, name=None):
    return th.sum(th.abs(var))


@cache_name_if_exist
def conv2d(input, filter, bias=None, strides=1, padding=0, name=None):
    shape = cg.get_shape(filter)
    H, W = shape[-2], shape[-1]
    padding = padding_str2tuple(padding, H, W)
    return F.conv2d(input, filter, bias, stride=tuple(strides), padding=padding)


@cache_name_if_exist
def inner_product(input, W, bias=None, name=None):
    if bias is not None:
        return th.addmm(bias, input, W)
    else:
        return input.matmul(W)


@cache_name_if_exist
def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None):
    assert len(ksize) == 4, "Only ksize of dim 4 is supported"
    padding = padding_str2tuple(padding, ksize[1], ksize[2])
    # The format of torch is two tuple ksize instead of 4.
    _ksize = []
    _ksize.append(ksize[1])
    _ksize.append(ksize[2])
    return F.max_pool2d(value, _ksize, strides, padding)


@cache_name_if_exist
def relu(v):
    return F.relu(v)


@cache_name_if_exist
def zero_fraction(data, name=None):
    return th.mean((data == 0).float())


@cache_name_if_exist
def mse_loss(data, labels, size_average=True, name=None):
    return th.nn.functional.mse_loss(data, labels, size_average=size_average)


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
