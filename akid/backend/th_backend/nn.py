from __future__ import division


import torch as th
from torch.nn import functional as F
from torch import nn as tnn

from .computational_graph import cache_name_if_exist
from . import computational_graph as cg
from akid.utils import glog as log


@cache_name_if_exist
def l2_loss(var, name=None):
    return th.div(th.sum(var * var), 2)


@cache_name_if_exist
def l2_norm(var, name=None):
    """
    Args:
         var: tensor
             tensor of rank 1 or 2. If the rank is 1, it is taken as a vector,
             and norm computed. If 2, it is taken as an array of vectors, where
             var[0] is a vector. The norm of each individually is computed.
    """
    shape = cg.get_shape(var)
    if len(shape) == 1:
        return th.sqrt(th.sum(var * var))
    elif len(shape) == 2:
        return th.sqrt(th.sum(var * var, dim=1))
    else:
        raise ValueError("Shape should or either 1 or 2. Got {}".format(len(shape)))


@cache_name_if_exist
def l1_loss(var, name=None):
    return th.sum(th.abs(var))


@cache_name_if_exist
def conv2d(input, filter, bias=None, strides=1, padding=0, name=None):
    shape = cg.get_shape(filter)
    H, W = shape[-2], shape[-1]
    shape = cg.get_shape(input)
    H_in, W_in = shape[-2], shape[-1]
    if type(padding) is str:
        padding = padding_str2tuple(H_in, W_in, strides, padding, H, W)
    strides = _normalize_stride(strides)
    return F.conv2d(input, filter, bias, stride=strides, padding=padding)


@cache_name_if_exist
def inner_product(input, W, bias=None, name=None):
    if bias is not None:
        return th.addmm(bias, input, W)
    else:
        return input.matmul(W)


@cache_name_if_exist
def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None):
    H, W = ksize[0], ksize[1]
    shape = cg.get_shape(value)
    H_in, W_in = shape[-2], shape[-1]
    padding = padding_str2tuple(H_in, W_in, strides, padding, H, W)
    strides = _normalize_stride(strides)
    return F.max_pool2d(value, ksize, strides, padding)


def _normalize_stride(strides):
    # The format of ksize in torch is a tuple of size 2 instead of 4 in
    # tensorflow.
    if len(strides) == 4:
        log.warning("Torch backend does not support stride in all four dimensions."
                    " Use the last two as height and width.")
        strides=(strides[-2], strides[-1])
    elif type(strides) is not tuple:
        strides = tuple(strides)

    return strides


@cache_name_if_exist
def avg_pool(value, ksize, strides, padding, name=None):
    H, W = ksize[0], ksize[1]
    shape = cg.get_shape(value)
    H_in, W_in = shape[-2], shape[-1]
    padding = padding_str2tuple(H_in, W_in, strides, padding, H, W)
    strides = _normalize_stride(strides)
    return F.avg_pool2d(value, ksize, strides, padding)


@cache_name_if_exist
def relu(v, name=None):
    return F.relu(v)


@cache_name_if_exist
def dropout(v, keep_prob, val=False, in_place=False, name=None):
    return F.dropout(v, 1-keep_prob, training=not val, inplace=in_place)


@cache_name_if_exist
def zero_fraction(data, name=None):
    return th.mean((data == 0).float())


@cache_name_if_exist
def mse_loss(data, labels, size_average=True, name=None):
    return th.nn.functional.mse_loss(data, labels, size_average=size_average)


@cache_name_if_exist
def cross_entropy_loss(logits, labels, name=None):
    return F.cross_entropy(logits, labels)


@cache_name_if_exist
def class_acccuracy(predictions, labels, name=None):
    size = cg.get_shape(labels)[0]
    pred = predictions.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct = pred.eq(labels.view_as(pred)).sum()
    acc = correct.float() / size
    return acc


@cache_name_if_exist
def normalize_weight(W):
    """
    Normalize the norm of each weight that corresponds to an output channel to 1.
    """
    shape = cg.get_shape(W)
    if len(shape) == 4:
        c_out, c_in, h, w = shape
        W = W.view(c_out, c_in*h*w)

    norm = l2_norm(W)
    W_normalized = W / norm[:, None]

    if len(shape) == 4:
        W_normalized = W_normalized.view(shape)

    return W_normalized


@cache_name_if_exist
def nn_riemannic_metric(K, W, b, need_to_normalize_weights=False):
    """
    Given Riemannian metric of the previous layer, compute that of this layer.

    Due to the fact that computing similarity between group action generated
    filters would require backtracking computation of similarity of all spatial
    location in the image, which is not feasible, only similarity between
    filters of different channels are used.

    Args:
        K: the Riemannian metric
        W: the weigth matrix, 4D or 2D, or None (means identity matrix)
        b: the bias vector
    """
    shape = cg.get_shape(W)

    if len(shape) == 4:
        c_out, c_in, h, w = shape
        W = W.permute(2, 3, 0, 1).contiguous()
        W = W.view(h*w, c_out, c_in)
        W_T = th.transpose(W, 1, 2)
        # Compute sum_{H * W}(WKW^{T}_{i})
        if K is not None:
            left = th.matmul(W, K)
        else:
            left = W
        if b is not None:
            mat = th.ger(b, b)  # Outer product
            K_out = th.addbmm(mat, left, W_T)
        else:
            K_out = th.sum(th.bmm(left, W_T), dim=0)

    elif len(shape) == 2:
        if K is not None:
            right = th.matmul(K, W)
        else:
            right = W

        if b is not None:
            mat = th.ger(b, b)  # Outer product
            K_out = th.addmm(mat, W.t(), right)
        else:
            K_out = th.mm(W.t(), right)
    else:
        raise ValueError("Shape {} is not supported".format(shape))

    K_out = K_out * (K_out > 0).float()

    return K_out


def padding_str2tuple(H_in, W_in, strides, padding, H, W):
    """
    Convert padding from string to tuple. Thea meaning of string is from
    tensorflow.
    """
    if padding == 'VALID':
        padding = 0
    elif padding == 'SAME':
        H_pad = get_padding_SAME(H_in, strides[0], H)
        W_pad = get_padding_SAME(W_in, strides[1], W)
        padding = (H_pad, W_pad)
    else:
        raise ValueError("{} padding is not supported. Should be VALID or SAME".format(padding))

    return padding


def get_padding_SAME(input_size, stride, ksize):
    out_size = (input_size + stride - 1) // stride
    padding_needed = max(0, (out_size - 1) * stride + ksize - input_size)
    padding_before = padding_needed // 2
    padding_after = padding_needed - padding_before
    # Since torch seems not to support 4 tuple padding, just return the left part
    padding = padding_before

    return padding
