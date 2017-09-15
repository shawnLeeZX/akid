"""
Akid aims to focus on enforcing the **way** to **build** entities/blocks that
process signal, instead of letting how signals/tensors propagates influence how
blocks are built. In this sense, the underlying tensor processing engine is
arbitrary, and abstracted as backends.
"""
import os

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
