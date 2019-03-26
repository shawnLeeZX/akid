"""
This module contains templates to build network blocks that consists of several
usual combination of layer types. For instance, a block of convolutional neural
network normally consists of a convolution layer, a pooling layer and an
activation layer. Those templates server as syntax sugars to
ease writing network structures. So instead of appending three layers and
organizing their in a statement block, one could just use a template to create
a block.
"""
from __future__ import absolute_import
from __future__ import print_function
import sys
import inspect

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
              conv_padding="SAME",
              initial_bias_value=0.,
              init_para={"name": "truncated_normal",
                         "stddev": 0.1},
              wd={"type": "l2", "scale": 5e-4},
              max_norm=None,
              in_channel_num=32,
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
                                      strides=[1, 1],
                                      padding=conv_padding,
                                      initial_bias_value=initial_bias_value,
                                      init_para=init_para,
                                      wd=wd,
                                      max_norm=max_norm,
                                      in_channel_num=in_channel_num,
                                      out_channel_num=out_channel_num,
                                      name="conv{}".format(counter)))
        if pool_size:
            block.append(PoolingLayer(ksize=pool_size,
                                      strides=pool_stride,
                                      padding="SAME",
                                      name="pool{}".format(counter)))
    else:
        block.append(InnerProductLayer(out_channel_num=out_channel_num,
                                       in_channel_num=in_channel_num,
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
            elif activation_type == "ngsmax":
                layer = SoftmaxNormalizationLayer(
                    name="ngsoftmax{}".format(counter),
                    group_size=activation["group_size"])
            elif activation_type == "gsmax":
                layer = GroupSoftmaxLayer(name="gsmax{}".format(counter),
                                          group_size=activation["group_size"])
            else:
                print(("{} activation type has not been supported"
                       " yet.".format(activation_type)))
                sys.exit(0)
        except KeyError as e:
                print(("{} not found. You perhaps have a typo or miss a"
                       " parameter.".format(e)))
                sys.exit(0)
    else:
        layer = None

    if activation_before_pooling:
        for i, b in enumerate(block):
            if type(b) is PoolingLayer:
                if layer:
                    block.insert(i, layer)
                if bn:
                    block.insert(i, bn_layer)
                break
    else:
        # Even if there is no activation layer, which happens at the last
        # readout layer, BN may still be needed.
        if bn:
            block.append(bn_layer)
        if layer:
            block.append(layer)

    if keep_prob:
        block.append(DropoutLayer(keep_prob=keep_prob,
                                  name="dropout{}".format(counter)))

    return block

__all__ = [name for name, x in locals().items() if not inspect.ismodule(x)]
