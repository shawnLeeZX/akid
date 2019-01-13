"""Operations often used for initializing tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import inspect

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops


def msra_initializer(factor=1.0, uniform=False, seed=None):
    """Returns an initializer that implement MSRA initialization. Refer to the
    [paper](http://arxiv.org/abs/1502.01852), *Delving Deep into Rectifiers:
    Surpassing Human-Level Performance on ImageNet Classification*, for
    details.

    It uses the version that keeps gradient variance 1. For the version to
    forward propagation variance 1, the `uniform_unit_scaling_initializer`
    provided by tensorflow suffices --- what you need to do is to change the
    multiplicative factor.

    Also, since the constant in MSRA init is also approximated in its
    derivation, a multiplication factor parameter is also introduced.

    Args:
        factor: Float.  A multiplicative factor by which the values will be
            scaled.
        seed: A Python integer. Used to create random seeds.

    Returns:
        An initializer that generates tensors with unit variance.
    """
    def _initializer(shape, dtype=dtypes.float32, partition_info=None):
        input_size = 1.0
        for i in shape[:-2]:
            input_size *= i
        input_size *= shape[-1]
        std = math.sqrt(2 / input_size) * factor
        if uniform:
            max_val = std
            return random_ops.random_uniform(shape, -max_val, max_val,
                                             dtype, seed=seed)
        else:
            return random_ops.truncated_normal(shape, stddev=std,
                                               dtype=dtype, seed=seed)
    return _initializer


__all__ = [name for name, x in locals().items() if not inspect.ismodule(x)]
