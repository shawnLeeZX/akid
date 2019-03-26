from __future__ import absolute_import
import inspect

from tensorflow.python.ops.image_ops import _Check3DImage
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf


def rescale_image(image):
    """Rescale `image` to range $[0, 1]$.

    It does this by dividing the largest value in the images.

    Args:
        image: 3-D tensor of shape `[height, width, channels]`.

    Returns:
        The rescaled image with same shape as `image`.

    Raises:
        ValueError: if the shape of 'image' is incompatible with this function.
    """
    _Check3DImage(image)

    image = math_ops.cast(image, dtype=dtypes.float32)
    image_max = tf.reduce_max(image)

    image = math_ops.div(image, image_max)
    return image


__all__ = [name for name, x in locals().items() if not inspect.ismodule(x)]
