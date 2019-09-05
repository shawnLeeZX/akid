from __future__ import division


from __future__ import absolute_import
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
def l2_norm(var, dim=None, name=None):
    """
    Args:
         var: tensor
             tensor of rank 1 or 2. If the rank is 1, it is taken as a vector,
             and norm computed. If 2, it is taken as an array of vectors, where
             var[0] is a vector. The norm of each individually is computed.
         dim: the dimension along which to compute l2 norm. Not useful when
             `var` is of one dimension.
    """
    shape = cg.get_shape(var)
    if len(shape) == 1 or dim is None:
        return th.sqrt(th.sum(var * var))
    else:
        return th.sqrt(th.sum(var * var, dim=dim))


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

    if len(padding) == 4:
        # We need to do manual padding since by default, conv2d only do
        # two-side padding with the same padding dim.
        padding = list(padding)
        padding.reverse()
        padding = tuple(padding)
        input = th.nn.functional.pad(input, padding)
        padding = 0

    return F.conv2d(input, filter, bias, stride=strides, padding=padding)


@cache_name_if_exist
def conv1d(x, W, b=None, stride=1, padding=0, name=None):
    return F.conv1d(x, W, b, stride, padding)


@cache_name_if_exist
def inner_product(input, W, bias=None, name=None):
    if bias is not None:
        return th.addmm(bias, input, W)
    else:
        return input.matmul(W)


@cache_name_if_exist
def bmm(M1, M2, name=None):
        return th.bmm(M1, M2)


@cache_name_if_exist
def max_pool(value, ksize, strides, padding, data_format="NHWC", return_indices=False, name=None):
    ksize = _normalize_ksize(ksize)
    H, W = ksize[0], ksize[1]
    shape = cg.get_shape(value)
    H_in, W_in = shape[-2], shape[-1]
    padding = padding_str2tuple(H_in, W_in, strides, padding, H, W)
    # Only take the padding at the beginning, since PyTorch by default only pad
    # the same at the both end.
    if type(padding) is not int and len(padding) == 4:
        padding = (padding[0], padding[2])
    strides = _normalize_stride(strides)
    return F.max_pool2d(value, ksize, strides, padding, return_indices=return_indices)


@cache_name_if_exist
def max_pool1d(x, ksize, stride, padding, name=None):
    return F.max_pool1d(x, ksize, stride, padding)


def _normalize_stride(strides):
    # The format of ksize in torch is a tuple of size 2 instead of 4 in
    # tensorflow.
    if type(strides) is int:
        strides = (strides, strides)
    elif len(strides) == 4:
        # log.warning("Torch backend does not support stride in all four dimensions."
        #             " Use the middle two as height and width.")
        strides=(strides[-3], strides[-2])
    elif type(strides) is not tuple:
        strides = tuple(strides)

    return strides

_normalize_ksize = _normalize_stride


@cache_name_if_exist
def avg_pool(value, ksize, strides, padding, name=None):
    ksize = _normalize_ksize(ksize)
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
def sigmoid(v, name=None):
    return th.sigmoid(v)


@cache_name_if_exist
def embedding(idxs, weights, name=None):
    return F.embedding(idxs, weights)


@cache_name_if_exist
def dropout(v, keep_prob, val=False, in_place=False, name=None):
    return F.dropout(v, 1-keep_prob, training=not val, inplace=in_place)


@cache_name_if_exist
def zero_fraction(data, name=None):
    return th.mean((data == 0).float())


@cache_name_if_exist
def mse_loss(data, labels, size_average=True, name=None):
    return th.nn.functional.mse_loss(data, labels, reduction="mean")


@cache_name_if_exist
def cross_entropy_loss(logits, labels, name=None):
    return F.cross_entropy(logits, labels)


@cache_name_if_exist
def margin_ranking_loss(x1, x2, target, margin=0, name=None):
    return F.margin_ranking_loss(x1, x2, target, margin=margin)


@cache_name_if_exist
def consine_similarity(x1, x2, dim=1, eps=1e-8):
    return F.cosine_similarity(x1, x2, dim, eps)


@cache_name_if_exist
def binary_cross_entropy_loss_with_logits(logits, labels, pos_weight=None, name=None):
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


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
    if type(strides) is int:
        strides = (strides, strides)

    if padding == 'VALID':
        padding = 0
    elif padding == 'SAME':
        H_pad = get_padding_SAME(H_in, strides[0], H)
        W_pad = get_padding_SAME(W_in, strides[1], W)
        if H_pad[0] == H_pad[1] and W_pad[0] == W_pad[1]:
            # Use plan to use the default padding in conv op.
            padding = (H_pad[0], W_pad[0])
        else:
            padding = (*H_pad, *W_pad)
    else:
        raise ValueError("{} padding is not supported. Should be VALID or SAME".format(padding))

    return padding


def get_padding_SAME(input_size, stride, ksize):
    out_size = (input_size + stride - 1) // stride
    padding_needed = max(0, (out_size - 1) * stride + ksize - input_size)
    padding_before = padding_needed // 2
    padding_after = padding_needed - padding_before
    # Since torch seems not to support 4 tuple padding, just return the left part
    padding = (padding_before, padding_after)

    return padding

@cache_name_if_exist
def bn(x, out_channel_num, # out_channel_num is unused.
       gamma, beta,
       running_mean=None, running_var=None,
       is_val=False, track_running_stats=True,
       exponential_average_factor=0.1, eps=1^-5,
       name=None, **kwargs):
    return F.batch_norm(
        x, running_mean, running_var, gamma, beta,
        not is_val or not track_running_stats,
        exponential_average_factor, eps)


@cache_name_if_exist
def hessian(l, x, name=None):
    """
    Compute the Hessian of l w.r.t. x, where l is the output of a loss function.
    """
    nabla = th.autograd.grad(l, x, create_graph=True)
    nabla = th.cat([g.contiguous().view(-1) for g in nabla])
    H = []
    for p in nabla:
        p = th.cat(th.autograd.grad(p, x, retain_graph=True))
        H.append(p)

    H = th.stack(H, 1)
    return H


@cache_name_if_exist
def grad(l, x, flatten=False, name=None):
    """
    If `flatten` is True, the output grad would be flattened to a vector.
    """
    nabla = th.autograd.grad(l, x, create_graph=True)

    if flatten:
        if type(nabla) is tuple:
            nabla = th.cat([g.contiguous().view(-1) for g in nabla])
        elif type(nabla) is th.Tensor:
            nabla = nabla.contiguous().view(-1)
        else:
            raise ValueError("Type {} of grad is not supported".format(type(nabla)))

    return nabla

@cache_name_if_exist
def hessian_vector_product(nabla, x, v, allow_unused=False, name=None):
    """
    Take a gradient `nabla` computed by PyTorch and compute the Hessian vector
    product between it and a vector `v`. To compute Hessian from `grad`, we
    need the parameters `x` that are supposed to be taken derivatives against.
    """
    Hv = th.autograd.grad(nabla, x, allow_unused=allow_unused, grad_outputs=v)
    # import ipdb; ipdb.set_trace()
    Hv = th.cat([p.contiguous().view(-1) for p in Hv])
    return Hv
