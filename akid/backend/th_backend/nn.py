from __future__ import division


import torch as th
from torch.nn import functional as F

from .computational_graph import cache_name_if_exist
from . import computational_graph as cg


@cache_name_if_exist
def l2_loss(var, name=None):
    return th.div(th.sum(var * var), 2)


@cache_name_if_exist
def l1_loss(var, name=None):
    return th.sum(th.abs(var))


@cache_name_if_exist
def conv2d(input, filter, bias=None, strides=1, padding=0, name=None):
    shape = cg.get_shape(filter)
    H, W = shape[-2], shape[-1]
    shape = cg.get_shape(input)
    H_in, W_in = shape[-2], shape[-1]
    padding = padding_str2tuple(H_in, W_in, strides, padding, H, W)
    return F.conv2d(input, filter, bias, stride=tuple(strides), padding=padding)


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
    # The format of ksize in torch is a tuple of size 2 instead of 4 in
    # tensorflow.
    return F.max_pool2d(value, ksize, tuple(strides), padding)


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
def nn_riemannic_metric(K, W, b):
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
