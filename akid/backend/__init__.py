"""
Akid aims to focus on enforcing the **way** to **build** entities/blocks that
process signal, instead of letting how signals/tensors propagates influence how
blocks are built. In this sense, the underlying tensor processing engine is
arbitrary, and abstracted as backends.
"""
from .tf_backend import *
from .tf_backend import nn
from .tf_backend import summary
