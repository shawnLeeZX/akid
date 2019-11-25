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

from .blocks import ProcessingLayer
from .systems import System, SequentialSystem, GraphSystem, SequentialGSystem
from .. import backend as A
from .common import (
    TRAIN_SUMMARY_COLLECTION,
    VALID_SUMMARY_COLLECTION,
)


class Brain(System, ProcessingLayer):
    """
    `Brain` supports variable sharing layer copy. By calling `get_copy` of the
    brain and call `forward` again, you will get a brain with the same
    configuration and that shares any variables original brain has.

    Note if `do_summary` and `moving_average_decay` are specified, it would
    override that option of any layers attached to this brain.
    """
    NAME = "Brain"

    def __init__(self, do_stat_on_norm=False, **kwargs):
        super(Brain, self).__init__(**kwargs)
        self.do_stat_on_norm = do_stat_on_norm

    def get_val_copy(self):
        """
        Get a copy for validation.
        """
        if A.backend() == A.TF:
            val_copy = self.get_copy()
        elif A.backend() == A.TORCH:
            val_copy = self
        else:
            raise ValueError("Not supported backend.")

        val_copy.set_val(True)

        return val_copy

    def get_shadow_copy(self):
        tower = self.get_copy()
        tower.set_shadow()

        return tower

    def set_val(self, val):
        """
        Change the state of the brain to validation.
        """
        assert type(val) is bool
        self.mode = A.Mode.VAL if val else A.Mode.TRAIN
        for b in self.blocks:
            b.set_val(val)

    def set_shadow(self):
        """
        Change the state of the brain to shadow replica.
        """
        super(Brain, self).set_shadow()
        for b in self.blocks:
            b.set_shadow()

    def switch_batch_monitoring_mode(self):
        """
        Switch on and off batch monitoring mode.
        """
        if self.is_mon:
            self.name = self.old_name
            self.set_flag("is_mon", False)
        else:
            # Save the old name and change status flag
            self.old_name = self.name
            self.name += "_mon"
            self.set_flag("is_mon", True)


    def set_do_summary_on_val_flag(self, v):
        super(Brain, self).set_do_summary_on_val_flag(v)
        for b in self.blocks:
            b.do_summary_on_val = v

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
                if v.requires_grad:
                    filter_list.append(v)

        return filter_list

    def num_parameters(self):
        total_parameters = 0
        for variable in self.get_filters():
            shape = A.get_shape(variable)
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim
            total_parameters += variable_parametes

        return total_parameters

    def _pre_setup(self):
        super(Brain, self)._pre_setup()
        if self.is_val:
            self.log("Setting up val brain {} ...".format(self.name))
        else:
            self.log("Setting up brain {} ...".format(self.name))

    def _pre_forward(self, *args, **kwargs):
        super(Brain, self)._pre_forward(*args, **kwargs)
        self.log("Forwarding in {} mode.".format(self.mode), debug=True)

    def _post_forward(self, *args, **kwargs):
        self._gather_loss()
        self._gather_evals()
        self._gather_train_ops()

        super(Brain, self)._post_forward(*args, **kwargs)

    def _gather_loss(self):
        """
        Gather all losses in all blocks in this brain.
        """
        loss_list = []
        for b in self.blocks:
            if b.loss is not None:
                loss_list.append(b.loss)
        # The total loss is defined as the cross entropy loss plus all of the
        # weight decay terms (L2 loss).
        self._loss = A.add_n(loss_list, name='total_loss')

    def _gather_evals(self):
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

        if self.is_val:
            verbose_eval_list = []
            for b in self.blocks:
                if b.verbose_eval is not None:
                    if type(b.verbose_eval) is list:
                        verbose_eval_list.extend(b.verbose_eval)
                    else:
                        verbose_eval_list.append(b.verbose_eval)

            if len(verbose_eval_list) > 0:
                self._verbose_eval = verbose_eval_list
            else:
                self._verbose_eval = None

    def _gather_train_ops(self):
        """
        Gather all train ops in all blocks in this brain.

        `self.train_op` points to a list.
        """
        # TODO: this method has a bad name, it is for some auxiliary ops that
        # should be run alongside with train_op in tensorflow.
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

    def attach(self, *block_in):
        for b in block_in:
            t = type(b)
            if t is list or t is tuple:
                for l in b:
                    self.attach(l)
            else:
                # A brain should only contain data processing layers.
                assert issubclass(t, ProcessingLayer), \
                    "A `Brain` should only contain `ProcessingLayer`s."
                super(Brain, self).attach(b)
                # Pass options down.
                if self.moving_average_decay:
                    # Only pass it down when it is not None.
                    b.moving_average_decay = self.moving_average_decay
                b.do_stat_on_norm = self.do_stat_on_norm


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
