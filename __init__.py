from __future__ import print_function

import os
import sys
import inspect

from akid.ops import image_ops as image

from akid.core.sources import *
from akid.datasets import *
from akid.core.kongfus import *
from akid.core.kids import *
from akid.core.sensors import *
from akid.core.jokers import *
from akid.core.common import *
from akid.core.brains import *
from akid.core import common
try:
    from akid.core.observer import *
except ImportError as e:
    print("Cannot import observer. You probably run on a machine without"
          " matplotlib.")


# Alert if AKID_DATA_PATH is not defined.
AKID_DATA_PATH = os.getenv("AKID_DATA_PATH")
if not AKID_DATA_PATH:
    print("Environment variable AKID_DATA_PATH is not defined. It is needed to"
          " run examples.", file=sys.stderr)

__all__ = [name for name, x in locals().items() if
           not inspect.ismodule(x) and not inspect.isabstract(x)]
