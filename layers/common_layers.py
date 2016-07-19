import inspect

import tensorflow as tf

from ..core.blocks import ProcessingLayer
from ..core.jokers import Joker


class ReshapeLayer(ProcessingLayer):
    """
    Reshape data. Only intrinsic shape information of output data should be
    given. The dimension of batch size is not needed. If no shape is given, all
    dimensions beyond batch size dimension are flattened.
    """
    def __init__(self, shape=None, **kwargs):
        # Do not do summary since we just reshape the data.
        kwargs["do_summary"] = False
        if "name" not in kwargs:
            kwargs["name"] = "reshape"
        super(ReshapeLayer, self).__init__(**kwargs)
        self.intrinsic_shape = shape

    def _setup(self, input):
        batch_size = input.get_shape().as_list()[0]
        if self.intrinsic_shape:
            shape = list(self.intrinsic_shape)
        else:
            shape = [-1]
        shape.insert(0, batch_size)
        self._data = tf.reshape(input, shape)


class PaddingLayer(ProcessingLayer, Joker):
    """
    Zero padding on height and width dimensions of the input feature map.

    This layer can work with input shape [H, W, C] and [N, H, W, C].
    """
    def __init__(self, padding=[1, 1], **kwargs):
        """
        Args:
            padding: a two-element list
                [H, W]. H holds how many padding will be added in height;
                similar W is for width. Padding is added symmetrically for each
                dimension.
        """
        # Do not do summary since we just pad the data.
        kwargs["do_summary"] = False
        if "name" not in kwargs:
            kwargs["name"] = "reshape"
        super(PaddingLayer, self).__init__(**kwargs)
        self.padding = padding

    def _setup(self, input):
        shape = input.get_shape().as_list()
        if len(shape) is 4:
            _padding = [
                [0, 0],
                [self.padding[0], self.padding[0]],
                [self.padding[1], self.padding[1]],
                [0, 0]
            ]
        else:
            _padding = [
                [self.padding[0], self.padding[0]],
                [self.padding[1], self.padding[1]],
                [0, 0]
            ]
        self._data = tf.pad(input, paddings=_padding)


__all__ = [name for name, x in locals().items() if not inspect.ismodule(x)]
