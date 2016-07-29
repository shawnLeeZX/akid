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


class GraphSystem(LinkedSystem):
    """
    A system that is capable to handle arbitrary graph style connections
    between blocks.

    It is supposed to contain `ProcessingLayer` that has a `inputs` attributes
    to hold interconnection information between layers.  If `inputs` is None,
    it means this layer is supposed to take the first output of previous layer
    (depending on the actually topology of the system this block is in) as its
    input data. If not None, a list of tuples should be passed in. For example,
    the input of this layer is supposed to be outputs of layer "conv_1" and
    "conv2", then a list of [{"name": "conv1", "idxs": [0]}, {"name":
    "conv2", "idxs": [0]}] should be passed in. Output of any blocks are a
    tuple (if there are multiple outputs). The list of indices means the
    indices of outputs of that layer to use.

    NOTE: no matter how the inputs are specified (by `inputs` or not), in the
    `_setup` method of a block, inputs feeds to a block is a tensor (if there
    is only one input) or a list of tensors (if there are multiple inputs.)
    """
    def _link_blocks(self, data_in):
        """
        Method overrode to handle arbitrary layer interconnections.
        """
        # Normalize input to a list for convenience even if there is only one
        # input.
        data = data_in if type(data_in) is list else [data_in]
        log.info("System input shape: {}".format(
            [d.get_shape().as_list() for d in data]))

        for l in self.blocks:
            log.info("Setting up block {}.".format(l.name))
            l.do_summary = self.do_summary
            inputs = None
            if l.inputs:
                # Find inputs in the system to current block.
                inputs = []
                for input in l.inputs:
                    # First check whether the input is from the system input.
                    if input["name"] == "system_in":
                        for i in input["idxs"]:
                            inputs.append(data_in[i])
                    # Then look through outputs of setup layers.
                    for b in self.blocks:
                        if b.is_setup and b.name == input["name"]:
                            # If a layer has only one output, directly put that
                            # data in the input since otherwise, this layer
                            # won't be listed at all.
                            if type(b.data) is not list:
                                inputs.append(b.data)
                            else:
                                for i in input["idxs"]:
                                    inputs.append(b.data[i])
                            break
                if len(inputs) is 1:
                    l.setup(inputs[0])
                else:
                    l.setup(inputs)
            else:
                l.setup(data[0])

            # Logging
            in_name = data[0].name if not inputs else [i.name for i in inputs]
            if l.data is not None:
                log.info("Connected: {} -> {}".format(
                    in_name,
                    l.data.name if type(l.data) is not tuple
                    else [d.name for d in l.data]))
                log.info("Top shape: {}".format(
                    l.data.get_shape().as_list() if l.data is not tuple
                    else [d.get_shape().as_list() for d in l.data]))
            else:
                log.info("Inputs: {}. No outputs.".format(in_name))

            data = l.data if type(l.data) is tuple else [l.data]

        self._data = data
