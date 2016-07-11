"""
This module contains templates to build network blocks that consists of several
usual combination of layer types. For instance, a block of convolutional neural
network normally consists of a convolution layer, a pooling layer and an
activation layer. Those templates server as syntax sugars to
ease writing network structures. So instead of appending three layers and
organizing their in a statement block, one could just use a template to create
a block.
"""
import sys
import inspect

from ..utils import glog as log
from akid.layers import (
    ConvolutionLayer,
    PoolingLayer,
    InnerProductLayer,
    DropoutLayer,
    CollapseOutLayer,
    ReLULayer,
    SoftmaxNormalizationLayer,
    BatchNormalizationLayer,
    GroupSoftmaxLayer
)


# A counter to give number count to layer names
counter = 0


def init():
    """
    Call this to reset counter value if more than one networks are being built
    in a running instance.
    """
    global counter
    counter = 0


def cnn_block(ksize=None,
              initial_bias_value=0.,
              init_para={"name": "truncated_normal",
                         "stddev": 0.1},
              wd={"type": "l2", "scale": 5e-4},
              max_norm=None,
              out_channel_num=32,
              pool_size=[2, 2],
              pool_stride=[2, 2],
              activation={"type": "relu"},
              activation_before_pooling=False,
              keep_prob=None,
              bn=None):
    global counter
    counter += 1
    block = []

    if ksize:
        block.append(ConvolutionLayer(ksize=ksize,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      initial_bias_value=initial_bias_value,
                                      init_para=init_para,
                                      wd=wd,
                                      max_norm=max_norm,
                                      out_channel_num=out_channel_num,
                                      name="conv{}".format(counter)))
        if pool_size:
            pool_size.insert(0, 1)
            pool_size.append(1)
            pool_stride.insert(0, 1)
            pool_stride.append(1)
            block.append(PoolingLayer(ksize=pool_size,
                                      strides=pool_stride,
                                      padding="SAME",
                                      name="pool{}".format(counter)))
    else:
        block.append(InnerProductLayer(out_channel_num=out_channel_num,
                                       initial_bias_value=initial_bias_value,
                                       init_para=init_para,
                                       wd=wd,
                                       name="ip{}".format(counter)))
    if bn:
        bn_layer = BatchNormalizationLayer(
            name="bn{}".format(counter), **bn)

    if activation:
        try:
            activation_type = activation["type"]
            if activation_type == "relu":
                layer = ReLULayer(name="relu{}".format(counter))
            elif activation_type == "maxout":
                layer = CollapseOutLayer(name="maxout{}".format(counter),
                                    group_size=activation["group_size"])
            elif activation_type == "gsoftmax":
                layer = SoftmaxNormalizationLayer(
                    name="gsoftmax{}".format(counter),
                    group_size=activation["group_size"])
            elif activation_type == "linearize":
                layer = GroupSoftmaxLayer(name="gsmax{}".format(counter),
                                          group_size=activation["group_size"])
            else:
                log.error("{} activation type has not been supported"
                          " yet.".format(activation_type))
                sys.exit(0)
        except KeyError as e:
            log.error("{} not found. You perhaps have a typo or miss a"
                      " parameter.".format(e))
            sys.exit(0)

        if activation_before_pooling:
            for i, b in enumerate(block):
                if type(b) is PoolingLayer:
                    block.insert(i, layer)
                    if bn:
                        block.insert(i, bn_layer)
                    break
        else:
            if bn:
                block.append(bn_layer)
            block.append(layer)
    else:
        # Even if there is no activation layer, which happens at the last
        # readout layer, BN may still be needed.
        if bn:
            block.append(bn_layer)

    if keep_prob:
        block.append(DropoutLayer(keep_prob=keep_prob,
                                  name="dropout{}".format(counter)))

    return block

__all__ = [name for name, x in locals().items() if not inspect.ismodule(x)]
