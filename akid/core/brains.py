"""
A brain is the data processing engine to process data supplied by `Sensor` to
fulfill certain tasks. More specifically,

* it builds up blocks to form an arbitrary network
* offers sub-graphs for inference, loss, evaluation, summaries
* provides access to all data and parameters within

To use a brain, feed in data as a list, as how it is done in any other
blocks. Some pre-specified brains are available under `akid.models.brains`. An
example that sets up a brain using existing brains is::

    # ... first get a feed sensor
    sensor.forward()
    brain = OneLayerBrain(name="brain")
    input = [sensor.data(), sensor.labels()]
    brain.forward(input)

Note in this case, `data()` and `labels()` of `sensor` returns tensors. It is
not always the case. If it does not, saying return a list of tensors, you need
do things like::

    input = [sensor.data()]
    input.extend(sensor.labels())

Act accordingly.

Similarly, all blocks work this way.

A brain provides easy ways to connect blocks. For example, a one layer brain
can be built through the following::

    class OneLayerBrain(GraphBrain):
        def __init__(self, **kwargs):
            super(OneLayerBrain, self).__init__(**kwargs)
            self.attach(
                ConvolutionLayer(ksize=[5, 5],
                                strides=[1, 1, 1, 1],
                                padding="SAME",
                                out_channel_num=32,
                                name="conv1")
            )
            self.attach(ReLULayer(name="relu1"))
            self.attach(
                PoolingLayer(ksize=[1, 5, 5, 1],
                            strides=[1, 5, 5, 1],
                            padding="SAME",
                            name="pool1")
            )

            self.attach(InnerProductLayer(out_channel_num=10, name="ip1"))
            self.attach(SoftmaxWithLossLayer(
                class_num=10,
                inputs=[
                    {"name": "ip1", "idxs": [0]},
                    {"name": "system_in", "idxs": [1]}],
                name="loss"))

It assembles a convolution layer, a ReLU Layer, a pooling layer, an inner
product layer and a loss layer. To attach a block (layer) that directly takes
the outputs of the previous attached layer as inputs, just directly attach the
block. If `inputs` exists, the brain will fetch corresponding tensors by name
of the block attached and indices of the outputs of that layer. See the loss
layer above for an example. Note that even though there are multiple inputs for
the brain, the first attached layer of the brain will take the first of these
input by default, given the convention that the first tensor is the data, and
the remaining tensors are normally labels, which is not used till very late.
"""
from __future__ import absolute_import, division, print_function

import inspect

import tensorflow as tf

from ..layers.synapse_layers import SynapseLayer
from .blocks import ProcessingLayer
from .systems import System, SequentialSystem, GraphSystem, SequentialGSystem


class Brain(System, ProcessingLayer):
    """
    `Brain` supports variable sharing layer copy. By calling `get_copy` of the
    brain and call `forward` again, you will get a brain with the same
    configuration and that shares any variables original brain has.

    Note if `do_summary` and `moving_average_decay` are specified, it would
    override that option of any layers attached to this brain.
    """
    def __init__(self, do_stat_on_norm=False, **kwargs):
        super(Brain, self).__init__(**kwargs)
        self.do_stat_on_norm = do_stat_on_norm

    def get_val_copy(self):
        """
        Get a copy for validation.
        """
        val_copy = self.get_copy()
        val_copy.set_val()
        return val_copy

    def get_shadow_copy(self):
        tower = self.get_copy()
        tower.set_shadow()

        return tower

    def set_val(self):
        """
        Change the state of the brain to validation.
        """
        self.is_val = True
        for b in self.blocks:
            b.is_val = True

    def set_shadow(self):
        """
        Change the state of the brain to shadow replica.
        """
        super(Brain, self).set_shadow()
        for b in self.blocks:
            b.set_shadow()

    def get_filters(self, names=None):
        """
        A public interface to expose filters of this brain.

        Args:
            names: a list of str
                If not None, only get filters in that list.

        Returns:
            A list: Filters are returned as a list, ordered by the order they
                are added to the brain.
        """
        filter_list = []
        for b in self.blocks:
            if names:
                if b.name not in names:
                    continue
            for v in b.var_list:
                filter_list.append(v)

        return filter_list

    def _pre_setup(self):
        super(Brain, self)._pre_setup()
        if self.is_val:
            self.log("Setting up val brain {} ...".format(self.name))
        else:
            self.log("Setting up brain {} ...".format(self.name))

    def _pre_forward(self, *args, **kwargs):
        super(Brain, self)._pre_forward(*args, **kwargs)
        self.log("Building forward propagation computational graph ...")

    def _post_forward(self, *args, **kwargs):
        super(Brain, self)._post_forward(*args, **kwargs)

        self._gather_loss_graphs()
        self._gather_eval_graphs()
        self._gather_train_ops()

        if self.do_summary:
            A.summary.scalar(self.loss.op.name, self.loss)

    def _gather_loss_graphs(self):
        """
        Gather all losses in all blocks in this brain.
        """
        loss_list = []
        for b in self.blocks:
            if b.loss is not None:
                loss_list.append(b.loss)
        # The total loss is defined as the cross entropy loss plus all of the
        # weight decay terms (L2 loss).
        self._loss = tf.add_n(loss_list, name='total_loss')

    def _gather_eval_graphs(self):
        """
        Gather all evaluation in all blocks in this brain.

        `self.eval` points to a list even there is only one evaluation metric.
        """
        eval_graph_list = []
        for b in self.blocks:
            if b.eval is not None:
                if type(b.eval) is list:
                    eval_graph_list.extend(b.eval)
                else:
                    eval_graph_list.append(b.eval)

        self._eval = eval_graph_list

    def _gather_train_ops(self):
        """
        Gather all train ops in all blocks in this brain.

        `self.train_op` points to a list.
        """
        train_op_list = []
        for b in self.blocks:
            if b.train_op is not None:
                if type(b.train_op) is list:
                    train_op_list.extend(b.train_op)
                else:
                    train_op_list.append(b.train_op)

        self._train_op = train_op_list

    def on_para_update(self):
        ops = []
        for b in self.blocks:
            ops.extend(b.on_para_update())

        return ops

    def attach(self, block_in):
        if type(block_in) is list:
            for l in block_in:
                l.do_summary_on_val = self.do_summary_on_val
                self.attach(l)
        else:
            # A brain should only contain data processing layers.
            assert issubclass(type(block_in), ProcessingLayer), \
                "A `Brain` should only contain `ProcessingLayer`s."
            block_in.do_summary_on_val = self.do_summary_on_val
            super(Brain, self).attach(block_in)
            # Pass options down.
            if self.moving_average_decay:
                # Only pass it down when it is not None.
                block_in.moving_average_decay = self.moving_average_decay
            block_in.do_stat_on_norm = self.do_stat_on_norm


class SequentialBrain(Brain, SequentialSystem):
    pass


class GraphBrain(Brain, GraphSystem):
    """
    A concrete class to build signal processing system (specifically neural
    network).

    To use the input data in a block using `inputs`, its name is
    "system_in". For example, say that the following layer uses the labels
    passed in::

        brain.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[{"name": "ip1", "idxs": [0]},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))
    """
    pass


class SeqentialGBrain(Brain, SequentialGSystem):
    pass


__all__ = [name for name, x in locals().items() if not inspect.ismodule(x)]
