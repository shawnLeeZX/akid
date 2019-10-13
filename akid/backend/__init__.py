"""
Akid aims to focus on enforcing the **way** to **build** entities/blocks that
process signal, instead of letting how signals/tensors propagates influence how
blocks are built. In this sense, the underlying tensor processing engine is
arbitrary, and abstracted as backends.

## Variable allocation

`akid` only distinguishes between variables that are trainable, and that are
not. To obtained a variable that is trainable, call `get_variable`; one that is
not, call `Tensor`. However, if any operations that involve both trainable and
non-trainable variables, the non-trainable should be declared with
`requires_grad` true, to indicate though they are not trainable, they should
save gradients (for back propagation).
"""
from __future__ import absolute_import
import os
import sys

import tensorflow as tf
import torch as th

from .computational_graph import *
from ..utils import glog
import gflags
FLAGS = gflags.FLAGS


if 'AKID_BACKEND' in os.environ:
    _backend = os.environ['AKID_BACKEND']
    assert _backend in {TF, TORCH}
    _BACKEND = _backend
else:
    _BACKEND = TF

if _BACKEND ==  TF:
    from .tf_backend import *
elif _BACKEND == TORCH:
    from .th_backend import *
else:
    raise ValueError("Backend {} is not supported.".format(_BACKEND))


def backend():
    """
    Public interface to get the backend in use.
    """
    return _BACKEND


def reset():
    reset_step()
    reset_block_count()
    close()
    # We use scope management from tensorflow, so reset graph
    tf.reset_default_graph()
    if _BACKEND == TORCH:
        summary.reset_collections()
        summary.close()


_log_initialized = False
gflags.DEFINE_string('f', '', 'kernel')

def init_log():
    global _log_initialized
    if not _log_initialized:
        FLAGS(sys.argv)
        glog.init("stdout", akid_logger=True)
        _log_initialized = True


def is_numerical(d):
    if isinstance(d, NamedNumericValue):
        return True

    if _BACKEND == TF:
        if isinstance(d, tf.Tensor):
            return True
    elif _BACKEND == TORCH:
        if isinstance(d, th.Tensor):
            return True

    return False
