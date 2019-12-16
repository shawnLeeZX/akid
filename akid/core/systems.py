"""
This module provides systems of different topology to compose `Block`s to
create more complex blocks. A system does not concern which type of block it
holds, but only concerns the mathematical topology how they connect.

Atrributes defined in a finer granularity overrides the same attributes defined
in a system. E.g., if `do_summary` is defined as `False` in a block, and the
block is attached to a system with `do_summary` valued `True`, the `False` will
take priority.
"""
from __future__ import absolute_import
import copy

from .blocks import GenerativeBlock
from .. import backend as A
from .interface_blocks import UpdateBlock


class System(GenerativeBlock, UpdateBlock):
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

    def __init__(self, do_stat_on_norm=False, **kwargs):
        super(System, self).__init__(**kwargs)

        self.do_stat_on_norm = do_stat_on_norm

        self.blocks = []

        # Quick lookup table to get the idx of a block. Since the blocks are
        # added in a topological order normally, this table provides quick
        # topological order lookup.
        self.block_name_to_idx = {}

    def get_copy(self):
        self_copy = super(System, self).get_copy()

        self_copy.blocks = []
        for b in self.blocks:
            self_copy.blocks.append(b.get_copy())

        return self_copy

    def get_clone(self):
        clone = super(System, self).get_copy()

        clone.blocks = []
        for b in self.blocks:
            clone.blocks.append(b.get_clone())

        return clone

    def set_shadow(self):
        super(System, self).set_shadow()
        for b in self.blocks:
            b.set_shadow()

    def get_last_layer_name(self):
        """
        Return the name of the last layer that is attached to the system. It is
        useful for getting the name and feed it to another layer as its input.
        """
        return self.blocks[-1].name

    def get_layer_by_name(self, name):
        """
        Get any layers in the system by its name. None if the layer is not
        found.
        """
        for b in self.blocks:
            if b.name is name:
                return b

        raise Exception("Layer {} is not found".format(name))

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

    def attach(self, block_in):
        # Set flags value to the value of this system if they are not
        # individually set.
        for f in self.__class__._flags:
            if getattr(block_in, f) is None:
                block_in.set_flag(f, getattr(self, f))

        self.blocks.append(block_in)

        self.block_name_to_idx[block_in.name] = len(self.blocks) - 1

    def set_do_summary_flag(self, v):
        super(System, self).set_do_summary_flag(v)
        for b in self.blocks:
            b.do_summary = v

    def set_flag(self, flag_name, v):
        """
        Set a flag that all blocks in the system all have recursively.
        """
        if flag_name in self.__class__._flags:
            setattr(self, flag_name, v)
        else:
            raise ValueError("{} does not have flag {}".format(self.__class__, flag_name))

        for b in self.blocks:
            b.set_flag(flag_name, v)


class SequentialSystem(System):
    """
    A system that links blocks one by one sequentially, like the linked list
    structure in Computer Science Data Structure.
    """
    def _setup(self):
        for b in self.blocks:
            self.log("Setting up block {}.".format(b.name))
            b.setup()

    def _forward(self, data_in):
        """
        Link the blocks linearly together. It takes exact one argument and
        apply processing blocks to it one by one, and return the final
        processed data.

        A `SequentialSystem` could be used standalone. However, another typical use
        of `SequentialSystem` is inherit it to make something more complex, such as
        creating a `Brain` sequentially linking layers together. In that case,
        `_forward` and `backward` would be overrided.
        """
        data = data_in
        try:
            shape = A.get_shape(data)
        except ValueError:
            shape = None

        if not self.done_first_pass:
            self.log("Doing forward propagation.")
            self.log("System input shape: {}".format(shape))

        previous_data_name = 'system_in'
        for l in self.blocks:

            data_out = l.forward(data)

            if not self.done_first_pass:
                name = A.get_name(data)
                p_name = name if name else previous_data_name
                name = A.get_name(data_out)
                n_name = name if name else l.name
                self.log("Connected: {} -> {}".format(p_name, n_name))
                previous_data_name = l.name

                if shape:
                    self.log("Top shape: {}".format(A.get_shape(data_out)))
            data = l.data

        self._data = data

        return data_out

    def _on_update(self):
        l = self.blocks[0]
        # The Riemannian metric is None for the first layer
        l._on_update(None)

        l_prev = l
        for l in self.blocks[1:]:
            l._on_update(l_prev.K)
            l_prev = l


class GraphSystem(SequentialSystem):
    """
    A system that is capable to handle arbitrary graph style connections
    between blocks.

    It is supposed to contain `ProcessingLayer` that has a `inputs` attributes
    to hold interconnection information between layers.  If `inputs` is None,
    it means this layer is supposed to take all the outputs of previous layer
    (depending on the actually topology of the system this block is in) as its
    input data. A tensor is the input if the previous layer only has one
    output, otherwise, a list of tensor would be the input. The first block of
    the system is special, see docstring of `Brain`. If not None, a list of
    tuples should be passed in. For example, the input of this layer is
    supposed to be outputs of layer "conv_1" and "conv2", then a list of
    [{"name": "conv1", "idxs": [0]}, {"name": "conv2", "idxs": [0]}] should be
    passed in. Output of any blocks are a tuple (if there are multiple
    outputs). The list of indices means the indices of outputs of that layer to
    use.

    It also supports directly pass tensor in.

    NOTE: no matter how the inputs are specified (by `inputs` or not), in the
    `_forward` method of a block, inputs feeds to a block is a tensor (if there
    is only one input) or a list of tensors (if there are multiple inputs.)
    """
    def _forward(self, data_in, injected_data=None, start_block=None, end_block=None):
        """
        Method overrode to handle arbitrary layer interconnections.

        It also supports partial propagation by specifying the names of the
        start and the end blocks. Note that in this case, `data_in` should be
        given in the shape as that of the input of `start_block` block.

        Args:
            data_in: list of tensors
            injected_data: dict
                A dict consists of injected data to replace the inputs that are
                needed in partial propagation.
            start_block: str
                The name of the start block for partial propagation.
            end_block: str
                The name of the end block for partial propagation. If None, and
                `start_block` is given, the system do forward propagation until
                the end.
        """
        # Normalize input to a list for convenience even if there is only one
        # input.
        t = type(data_in)
        data = data_in if t is list or t is tuple else [data_in]

        if not self.done_first_pass:
            self.log("System input shape: {}".format(
                [A.get_shape(d) for d in data]))

        previous_data_name = "system_in"

        # Check if we do partial propagation.
        if start_block is None:
            start_block = self.blocks[0].name

        # Find the start_block and start from it.
        for start_idx,l in enumerate(self.blocks):
            if l.name == start_block:
                break

        if end_block is None:
            end_block = self.blocks[-1].name

        # Find the end block if given.
        for end_idx in range(1, len(self.blocks)+1):
            if self.blocks[-end_idx].name == end_block:
                end_idx = len(self.blocks) - end_idx + 1
                break

        if start_idx > end_idx:
            raise ValueError("Start idx should be larger than the end idx.")

        for i, l in enumerate(self.blocks[start_idx:end_idx]):

            inputs = None
            if l.inputs:
                # Find inputs in the system to current block.
                inputs = []
                for input in l.inputs:
                    if A.is_tensor(input):
                        inputs.append(input)
                        continue

                    input_num = len(inputs)

                    # Pick the first output tensor, if not specifying idxs
                    if "idxs" not in input:
                        input["idxs"] = [0]

                    # First check whether the input is from the system input.
                    if input["name"] == "system_in":
                        for i in input["idxs"]:
                            inputs.append(data_in[i])
                    # Then look through outputs of setup layers.
                    else:
                        block_idx = self.block_name_to_idx[input["name"]]
                        b = self.blocks[block_idx]
                        data = b.data

                        if block_idx < start_idx:
                            # The input should be fed in from system_in.
                            data = injected_data[input["name"]]

                        # If a layer has only one output, directly put
                        # that data in the input since otherwise, this
                        # layer won't be listed at all.
                        if type(data) is not list:
                            inputs.append(data)
                        else:
                            for i in input["idxs"]:
                                inputs.append(data[i])

                    if "idxs" not in input:
                        input_inc = 1
                    else:
                        input_inc = len(input["idxs"])
                    if len(inputs) != input_num + input_inc:
                        raise Exception("{} is not found. You perhaps misspell"
                                        " the layer name.".format(
                                            input["name"]))

                if len(inputs) is 1:
                    l.forward(inputs[0])
                else:
                    l.forward(inputs)
            else:
                # By default, we only pass the first tensor to the new layer.
                l.forward(data[0])

            if not self.done_first_pass:
                # Logging
                dtype = type(data)
                if inputs:
                    in_name = [A.get_name(i) for i in inputs]
                    if None in in_name:
                        # No name is created for the inputs, thus, we use their
                        # shape instead.
                        in_name = [str(i.shape) for i in inputs]
                else:
                    in_name = A.get_name(data[0])

                if in_name is None:
                    in_name = previous_data_name

                if l.data is not None:
                    dtype = type(l.data)
                    if dtype is not tuple and dtype is not list:
                        n = A.get_name(l.data)
                        out_name = n if n else l.name
                    else:
                        out_name = [A.get_name(d) for d in l.data]
                        if None in out_name:
                            out_name = l.name

                    self.log("Connected: {} -> {}".format(in_name, out_name))
                    self.log("Top shape: {}".format(
                        A.get_shape(l.data) if dtype is not tuple and dtype is not list
                        else [A.get_shape(d) for d in l.data]))
                else:
                    self.log("Inputs: {}. No outputs.".format(in_name))

            dtype = type(l.data)
            if l.data is not None:
                data = l.data if dtype is tuple or dtype is list else [l.data]
            else:
                data = None

            previous_data_name = l.name

        self._data = data

        return data


class SequentialGSystem(GraphSystem):
    """
    A `System` that supports arbitrary connectivity forward propagation, but
    backward propagation only supports sequentially connected blocks. Roughly,
    it is a generative system that supports `GraphSystem` way of specifying
    labels.
    """
    def backward(self, X_in):
        """
        Call `backward` in each linked block in reverse order.

        NOTE: The sequentially linked version is written first given a topology
        order may need to be inferred if we want to do arbitrary connectivity.
        """
        # Assume only one input
        if type(X_in) is list:
            X_in = X_in[0]

        data = X_in
        try:
            shape = A.get_shape(data)
        except ValueError:
            shape = None

        self.log("Doing backward propagation.")
        self.log("System input shape: {}".format(shape))
        for l in reversed(self.blocks):
            l.backward(data)
            self.log("Connected: {} -> {}".format(data.name,
                                                  l.data_g.name))
            if shape:
                self.log("Top shape: {}".format(A.get_shape(l.data)))
            data = l.data_g

        self._data_g = data

        return self._data_g
