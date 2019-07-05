"""
This module contains `Joker`s, the classes that make benign jokes to data, aka
doing data augmentations.
"""
from __future__ import absolute_import, division, print_function

import abc
import sys
import inspect

import tensorflow as tf

from .blocks import ShadowableBlock
from .systems import SequentialSystem
import akid


class Joker(ShadowableBlock):
    """
    A top level abstract class to do data augmentations. Since it is also part
    of a data processing process, it is a sub-class of `ShadowableBlock`.

    A Joker normally accepts one input -- so its `_forward` only takes one input,
    and gives out one output -- so the output is revealed via property `data`.
    """
    def __init__(self, do_summary=False, **kwargs):
        """
        Args:
            do_summary: Boolean
                Do not to do summary on output by default, since it would add
                too much relatively repeated summary since the statistics of
                data won't change much.
        """
        # do_summary is argument of the super class. We change its default
        # value, so we put the changed value in.
        kwargs["do_summary"] = do_summary
        super(Joker, self).__init__(**kwargs)

    @abc.abstractmethod
    def _forward(self, data_in):
        raise NotImplementedError("Each `Joker` should implement `_forward` to"
                                  " do actual data augmentation!")
        sys.exit()

    @property
    def data(self):
        """
        All sub-class `Joker` should save the augmented data to `_data`.
        """
        return self._data


class JokerSystem(SequentialSystem):
    """
    A system consists of linearly linked jokers to do data augmentation.
    """
    def attach(self, joker):
        assert issubclass(type(joker), Joker),\
            "A `JokerSystem` should only contain `Joker`s."
        super(JokerSystem, self).attach(joker)


class CropJoker(Joker):
    def __init__(self,
                 height=None, width=None,
                 center=False, central_fraction=None,
                 **kwargs):
        """
        A `Joker` that randomly crop a region of `height` and `width` from
        an input image.

        Args:
            center: a Boolean. Whether to resize an image to a target width
                and height by either centrally cropping the image or padding it
                evenly with zeros, or randomly crop the input image to get an
                image with desired size.
            central_fraction: float
                If existing, crop the image by fraction instead of using height
                and width passed in.
        """
        super(CropJoker, self).__init__(**kwargs)
        self.height = height
        self.width = width
        self.center = center
        self.central_fraction = central_fraction

    def _forward(self, data_in):
        if self.center:
            self.log("Center crop images.")
            if self.central_fraction:
                self._data = tf.image.central_crop(
                    data_in, central_fraction=self.central_fraction)
            else:
                self._data = tf.image.resize_image_with_crop_or_pad(
                    data_in, self.height, self.width)
        else:
            self.log("Randomly crop images.")
            shape = data_in.get_shape().as_list()
            assert self.width and self.height,\
                "crop height and width should not be None."
            self._data = tf.random_crop(data_in,
                                        [self.height, self.width, shape[-1]])

        return self._data


class FlipJoker(Joker):
    def __init__(self, flip_left_right=True, **kwargs):
        """
        A `Joker` that randomly flips input images.

        Args:
            flip_left_right: Boolean
                If True, do randomly horizontal flipping, otherwise, do
                vertical flipping.
        """
        super(FlipJoker, self).__init__(**kwargs)
        self.flip_left_right = flip_left_right

    def _forward(self, data_in):
        if self.flip_left_right:
            self.log("Randomly flip image left right.")
            self._data = tf.image.random_flip_left_right(data_in)
        else:
            self.log("Randomly flip image up down.")
            self._data = tf.image.random_flip_up_down(data_in)

        return self._data


class LightJoker(Joker):
    def __init__(self, contrast=True, brightness=True, **kwargs):
        """
        A `Joker` that randomly adjust contrast and brightness of input images.

        Args:
            contrast: Boolean
                If True, randomly adjust contrast.
            brightness: Boolean
                If True, randomly adjust brightness.
        """
        super(LightJoker, self).__init__(**kwargs)
        self.contrast = contrast
        self.brightness = brightness

    def _forward(self, data_in):
        data = data_in
        # TODO(Shuai): The parameters should not be hard coded.
        if self.contrast:
            self.log("Randomly change contrast.")
            data = tf.image.random_contrast(data, lower=0.2, upper=1.8)
        if self.brightness:
            self.log("Randomly change brightness.")
            data = tf.image.random_brightness(data, max_delta=63)

        self._data = data

        return self._data


class WhitenJoker(Joker):
    """
    Per image whitening joke.
    """
    def _forward(self, data_in):
        self._data = tf.image.per_image_standardization(data_in)
        return self._data


class RescaleJoker(Joker):
    """
    Rescale images to $[0, 1]$.
    """
    def _forward(self, data_in):
        self._data = akid.image.rescale_image(data_in)
        return self._data


class ResizeJoker(Joker):
    def __init__(self,
                 height,
                 width,
                 resize_method=0,
                 **kwargs):
        """
        A `Joker` that resizes an image to [height, width].

        Args:
            resize_method: int
                the resize method to use, see doc of `tf.image.resize_images`
                for details. Briefly, available resize methods are (retrieved
                from tensorflow version 0.11):
                    BILINEAR = 0
                    NEAREST_NEIGHBOR = 1
                    BICUBIC = 2
                    AREA = 3
        """
        super(ResizeJoker, self).__init__(**kwargs)
        self.height = height
        self.width = width
        self.resize_method = resize_method

    def _forward(self, data_in):
        shape = data_in.get_shape().as_list()
        if len(shape) == 3:
            data = tf.expand_dims(data_in, 0)
        resized_image = tf.image.resize_images(data,
                                               [self.height, self.width],
                                               method=self.resize_method)
        resized_image = tf.squeeze(resized_image)
        self._data = resized_image

        return self._data


__all__ = [name for name, x in locals().items() if
           not inspect.ismodule(x) and not inspect.isabstract(x)]
