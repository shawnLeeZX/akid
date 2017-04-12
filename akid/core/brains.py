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
    sensor.setup()
    brain = OneLayerBrain(name="brain")
    input = [sensor.data(), sensor.labels()]
    brain.setup(input)

Note in this case, `data()` and `labels()` of `sensor` returns tensors. It is
not always the case. If it does not, saying return a list of tensors, you need
do things like::

    input = [sensor.data()]
    input.extend(sensor.labels())

Act accordingly.

Similarly, all blocks work this way.

A brain provides easy ways to connect blocks. For example, a one layer brain
can be built through the following::

    class OneLayerBrain(Brain):
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
from .systems import GraphSystem


class Brain(GraphSystem, ProcessingLayer):
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

    `Brain` supports variable sharing layer copy. By calling `get_copy` of the
    brain and call `setup` again, you will get a brain with the same
    configuration and that shares any variables original brain has.

    Note if `do_summary` and `moving_average_decay` are specified, it would
    override that option of any layers attached to this brain.
    """
    def __init__(self, do_stat_on_norm=False, **kwargs):
        """
        Note a `Brain` contains a tensorflow `Graph` class. It is used to build
        a graph when doing visualization. When visualization is factored out,
        the graph member should be deleted as well.

        TODO(Shuai): think a more elegant way to handle polymorphism
        To get around the issues that Brain cannot inherit constructors of both
        ancestor, the paras of `LinkedSystem` is re-coded here.
        """
        ProcessingLayer.__init__(self, **kwargs)
        self.blocks = []
        self.do_stat_on_norm = do_stat_on_norm

    def attach(self, block_in):
        """
        Attach a layer or a block to the brain.

        As for adding a layer, if it is an intermediate processing layer, it
        will be appended to previous layers; If it is a loss layer, well, it is
        added as a loss layer.

        As for adding a block, a block should be a list of layers. Layers in
        the list will be added by the order they appear in the list.

        For now only one loss layer at the end of a network is supported.

        Args:
            block_in: a `ProcessingLayer` or a list of it.

        """
        if type(block_in) is list:
            for l in block_in:
                self.attach(l)
        else:
            # A brain should only contain data processing layers.
            assert issubclass(type(block_in), ProcessingLayer), \
                "A `Brain` should only contain `ProcessingLayer`s."
            super(Brain, self).attach(block_in)
            # Pass options down.
            if self.moving_average_decay:
                # Only pass it down when it is not None.
                block_in.moving_average_decay = self.moving_average_decay
            block_in.do_stat_on_norm = self.do_stat_on_norm

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

    def _setup(self, data_in):
        """
        Set up the computational graph.

        Args:
            data_in: a list of tensors or placeholders
                Data supplied by `Sensor`, including labels.
        """
        if self.is_val:
            self.log("Setting up val brain {} ...".format(self.name))
        else:
            self.log("Setting up brain {} ...".format(self.name))
        self._setup_graph(data_in)
        self._gather_loss_graphs()
        self._gather_eval_graphs()
        self._gather_train_ops()

    def _setup_graph(self, data_in):
        """
        Build the net up to where it may be used for inference.
        """
        self._link_blocks(data_in)

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

    def _post_setup(self):
        if self.do_summary:
            tf.summary.scalar(self.loss.op.name, self.loss)

    def on_batch_finishes(self):
        # Max norm constrain.
        clipped_filters = []
        for b in self.blocks:
            if issubclass(type(b), SynapseLayer):
                if b.clipped_filters:
                    clipped_filters.extend(b.clipped_filters)

        if len(clipped_filters) is not 0:
            self.max_norm_clip_op = tf.group(*clipped_filters)
        else:
            self.max_norm_clip_op = None

__all__ = [name for name, x in locals().items() if
           not inspect.ismodule(x) and not inspect.isabstract(x)]
