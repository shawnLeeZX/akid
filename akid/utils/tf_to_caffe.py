"""
This module contains utilities to transform tensorflow style objects to Caffe
style.
"""

from __future__ import absolute_import
import sys
import numpy as np
from . import glog as log


def tf_to_caffe_blob(blob):
    """
    Transform tensorflow style numpy array to Caffe blob style.

    Args:
        blob: numpy.array
            tensorflow style numpy array.

    Return:
        Transformed numpy array.
    """
    if len(blob.shape) is 4:
        blob = np.einsum('hwcn->nchw', blob)
    elif len(blob.shape) is 2:
        blob = np.einsum('nc->cn', blob)
    elif len(blob.shape) is 3:
        blob = np.einsum('hwc->chw', blob)
    else:
        log.error("{} shape numpy array is not supported!".format(blob.shape))
        sys.exit()
    return blob
