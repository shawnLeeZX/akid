"""
This module contains network models.
"""
from __future__ import absolute_import, division, print_function

import copy
import inspect

import tensorflow as tf

from ..layers.loss_layers import LossLayer
from ..layers.synapse_layers import SynapseLayer
from .blocks import ProcessingLayer
from .systems import GraphSystem

from ..utils import glog as log


class Brain(GraphSystem, ProcessingLayer):
    """
    Class `Brain` is the data processing engine to process data supplied by
    `Sensor` to fulfill certain tasks. More specifically,
    * it builds up blocks to form a network
    * offers sub-graphs for inference, loss, evaluation
    * does statistical summary
    * visualization of data, filters and feature maps

    `Brain` supports variable sharing layer copy. By calling `get_copy` of the
    brain and call `setup` again, you will get a brain with the same
    configuration and that shares any variables original brain has.

    Note if `do_summary` and `moving_average_decay` are specified, it would
    override that option of any layers attached to this brain.

    For now, this net is a sequentially built, non-branch, one loss at the top
    layer network. Also, how stat summaries are gathered only works layers with
    one output for now.
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

    def get_filters(self):
        """
        A public interface to expose filters of this brain.

        Returns:
            A list: Filters are returned as a list, ordered by the order they
                are added to the brain.
        """
        filter_list = []
        for b in self.blocks:
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
            log.info("Setting up val brain {} ...".format(self.name))
        else:
            log.info("Setting up brain {} ...".format(self.name))
        self._setup_graph(data_in)
        self._gather_loss_graphs()
        self._gather_eval_graphs()

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

    def _post_setup(self):
        if self.do_summary:
            tf.scalar_summary(self.loss.op.name, self.loss)

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
