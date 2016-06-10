"""
This module provides systems of different topology to compose `Block`s to
create more complex blocks. A system does not concern which type of block it
holds, but only concerns the mathematical topology how they connect.
"""
import copy
import abc
import sys

from .blocks import Block
from ..utils import glog as log


class System(Block):
    """
    A top level class to model a system that is purposeless. It means this
    system does not serve a clear purpose, but an aggregation of blocks.

    If you want to make it of purpose, or in another word, enforce semantics on
    it, you could combine with other classes by multi-inheritance and override
    `attach` method to add a gate keeper. For instance, a `Brain` should only
    contain `ProcessingLayer` since its purpose is to processing data. So
    `Brain` could also be taken as a `ProcessingLayer` layer and it is also a
    system.

    It enforces an interface `data` to provide the data propagated in this
    system.

    A system has direct access to the outputs of the blocks it contains, but it
    only should use them when an action that cannot be accomplished without
    information from more than one blocks.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, do_stat_on_norm=False, **kwargs):
        super(System, self).__init__(**kwargs)

        self.do_stat_on_norm = do_stat_on_norm

    @abc.abstractmethod
    def data(self):
        raise NotImplementedError("Each system should implement the interface"
                                  " `data` to provide the data propagated in"
                                  " it to the outside world!")
        sys.exit()


class LinkedSystem(System):
    """
    A system that links blocks one by one sequentially, like the linked list
    structure in Computer Science Data Structure.
    """
    def __init__(self, **kwargs):
        super(LinkedSystem, self).__init__(**kwargs)

        self.blocks = []
        self.block_names = []

    def get_copy(self):
        self_copy = copy.copy(self)

        self_copy.blocks = []
        for b in self.blocks:
            self_copy.blocks.append(b.get_copy())

        return self_copy

    def attach(self, block_in):
        """
        Attach a block to the system.
        """
        self.blocks.append(block_in)

    @property
    def is_empty(self):
        if self.blocks:
            return False

        return True

    @property
    def data(self):
        """
        Since it only has one output, override method to a property.
        """
        return self._data

    def _setup(self, data_in):
        """
        A `LinkedSystem` could be used standalone. However, another typical use
        of `LinkedSystem` is inherit it to make something more complex, such as
        creating a `Brain` sequentially linking layers together. In that case,
        `_setup` would be overrided. So we move the linking operation to
        another private function so after sub-class of this class does not need
        to rewrite the linking code.
        """
        self._link_blocks(data_in)

    def _link_blocks(self, data_in):
        """
        Link the blocks linearly together. It takes exact one argument and
        apply processing blocks to it one by one, and return the final
        processed data.
        """
        data = data_in
        log.info("System input shape: {}".format(data.get_shape().as_list()))
        for l in self.blocks:
            log.info("Setting up block {}.".format(l.name))
            l.do_summary = self.do_summary
            l.setup(data)
            log.info("Connected: {} -> {}".format(data.name,
                                                  l.data.name))
            log.info("Top shape: {}".format(l.data.get_shape().as_list()))
            data = l.data

        self._data = data
