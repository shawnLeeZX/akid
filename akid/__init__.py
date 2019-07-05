from __future__ import print_function

from __future__ import absolute_import
import os
import sys
import inspect

from akid.ops import image_ops as image

from akid.core.sources import *
from akid import datasets
from akid.datasets import *
from akid.core.kids import *
from akid.core.kongfus import *
from akid.core.sensors import *
from akid.core.jokers import *
from akid.core.common import *
from akid.core.brains import *
from akid.core.eval_blocks import *
from akid.core import events
from akid.core import initializers
from akid.core import common
from akid.core import engines
from akid.core import samplers
try:
    from akid.core.observer import *
except ImportError as e:
    print("Cannot import observer. You probably run on a machine without"
          " matplotlib.")

# Expose sub-modules
import akid.nn as nn
import akid.layers as layers
from akid import backend as A


def reset():
    A.reset()
    reset_eval_block_map()


__all__ = [name for name, x in locals().items() if
           not inspect.ismodule(x) and not inspect.isabstract(x)]
