"""
The module provides samplers to iterate data in different sequence.
"""
from __future__ import print_function
from __future__ import absolute_import
from six import raise_from

import random

from ..utils import glog as log
from .events import EpochCompletedEvent
from six.moves import range


sampler_dict = {}


def get(sampler_name, *args, **kwargs):
    """
    Given the name of the sampler, return a Sampler that has a `next()`
    interface to iterate the dataset.
    """
    try:
        sampler = sampler_dict[sampler_name]
    except ValueError as e:
        raise_from(ValueError("Sampler {} is not supported".format(sampler_name)), e)

    return sampler.get(*args, **kwargs)


class SamplerRegistry(object):
    def __init__(self,
                 name,
                 class_name,
                 message=None):
        """
        Args:
            name: str:
                Name of the initializer. The unique identifier.
            class_name: class
                A Sampler class.
            message: str
                Usage help message.
        """
        self.message = message
        self.name = name
        self.class_name = class_name
        sampler_dict[name] = self

    def get(self, *args, **kwargs):
        return self.class_name(*args, **kwargs)

    def help(self):
        print(self.message)


class Sampler(object):
    """
    Given a list of indices, a `Sampler` provides a function `next` to return
    the next batch of indices in a certain order.
    """
    def __init__(self, length, *args, **kwargs):
        super(Sampler, self).__init__(*args, **kwargs)
        self.indices = list(range(length))
        self.current_idx = 0

    def reset(self):
        self.current_idx = 0

    def _check_epoch_finishes(self):
        if self.current_idx >= len(self.indices):
            self.reset()
            log.debug("An epoch finished. Reset.")
            raise EpochCompletedEvent()

class SequenceSampler(Sampler):
    def next(self, size=1):
        self._check_epoch_finishes()

        if self.current_idx + size > len(self.indices):
            index_batch = self.indices[self.current_idx:]
            self.current_idx = len(self.indices)
        else:
            index_batch = self.indices[self.current_idx:self.current_idx+size]
            self.current_idx += size

        return index_batch


class ShuffleSampler(SequenceSampler):
    def __init__(self, *args, **kwargs):
        super(ShuffleSampler, self).__init__(*args, **kwargs)
        random.shuffle(self.indices)

    def reset(self):
        super(ShuffleSampler, self).reset()
        random.shuffle(self.indices)


SamplerRegistry("shuffle",
                ShuffleSampler,
                message="Randomly shuffle the dataset for each epoch.")
SamplerRegistry("sequence",
                SequenceSampler,
                message="Sequentially sample the dataset for each epoch.")
