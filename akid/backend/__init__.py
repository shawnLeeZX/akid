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
import os

import tensorflow as tf

from .computational_graph import *


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
    reset_eval_block_map()
    close()
    # We use scope management from tensorflow, so reset graph
    tf.reset_default_graph()
    if _BACKEND == TORCH:
        summary.reset_collections()
        summary.close()
