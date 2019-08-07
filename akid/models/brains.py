from __future__ import absolute_import
from akid import GraphBrain
from akid.layers import (
    ConvolutionLayer,
    PoolingLayer,
    ReLULayer,
    InnerProductLayer,
    SoftmaxWithLossLayer,
    LRNLayer,
    BatchNormalizationLayer,
    DropoutLayer,
    MergeLayer,
    ReshapeLayer,
    PaddingLayer,
    CollapseOutLayer,
    GroupSoftmaxLayer,
    MaxPoolingLayer
)
from akid import backend as A
from six.moves import range


class AlexNet(GraphBrain):
    """
    A class for alex net specifically.
    """
    def __init__(self, in_channel_num, dataset="imagenet",  **kwargs):
        super(AlexNet, self).__init__(**kwargs)

        self.in_channel_num = in_channel_num

        if dataset == "imagenet":
            self._build_imagenet_model()
        else:
            self._build_cifar10_model()

    def _build_imagenet_model(self):
        wd={"type": "l2", "scale": 1e-4}
        init = {"name": "xavier"}
        self.attach(ConvolutionLayer(ksize=[11, 11],
                                     strides=[4, 4],
                                     padding="VALID",
                                     init_para=init,
                                     wd=wd,
                                     in_channel_num=3,
                                     out_channel_num=64,
                                     name='conv1'))
        self.attach(ReLULayer(name='relu1'))
        self.attach(MaxPoolingLayer(ksize=[3, 3],
                                    strides=[2, 2],
                                    padding="VALID",
                                    name="pool1"))
        self.attach(ConvolutionLayer(ksize=[5, 5],
                                     strides=[1, 1],
                                     padding="SAME",
                                     init_para=init,
                                     wd=wd,
                                     in_channel_num=64,
                                     out_channel_num=192,
                                     name='conv2'))
        self.attach(ReLULayer(name='relu2'))
        self.attach(MaxPoolingLayer(ksize=[3, 3],
                                    strides=[2, 2],
                                    padding="VALID",
                                    name="pool2"))
        self.attach(ConvolutionLayer(ksize=[3, 3],
                                     strides=[1, 1],
                                     padding="SAME",
                                     init_para=init,
                                     wd=wd,
                                     in_channel_num=192,
                                     out_channel_num=384,
                                     name='conv3'))
        self.attach(ReLULayer(name='relu3'))
        self.attach(ConvolutionLayer(ksize=[3, 3],
                                     strides=[1, 1],
                                     padding="SAME",
                                     init_para=init,
                                     wd=wd,
                                     in_channel_num=384,
                                     out_channel_num=256,
                                     name='conv4'))
        self.attach(ReLULayer(name='relu4'))
        self.attach(ConvolutionLayer(ksize=[3, 3],
                                     strides=[1, 1],
                                     padding="SAME",
                                     init_para=init,
                                     wd=wd,
                                     in_channel_num=256,
                                     out_channel_num=256,
                                     name='conv5'))
        self.attach(ReLULayer(name='relu5'))
        self.attach(MaxPoolingLayer(ksize=[3, 3],
                                    strides=[2, 2],
                                    padding="VALID",
                                    name="pool2"))
        self.attach(ReshapeLayer(name='reshape'))
        self.attach(DropoutLayer(keep_prob=0.5, name="dropout1"))
        self.attach(InnerProductLayer(in_channel_num=256*5*5,
                                      out_channel_num=4096,
                                     init_para=init,
                                      wd=wd,
                                      name='ip1'))
        self.attach(ReLULayer(name='relu_ip1'))
        self.attach(DropoutLayer(keep_prob=0.5, name="dropout2"))
        self.attach(InnerProductLayer(in_channel_num=4096,
                                      out_channel_num=4096,
                                     init_para=init,
                                      wd=wd,
                                      name='ip2'))
        self.attach(ReLULayer(name='relu_ip2'))
        self.attach(InnerProductLayer(in_channel_num=4096,
                                      out_channel_num=self.in_channel_num,
                                     init_para=init,
                                      wd=wd,
                                      name='ip_last'))
        self.attach(SoftmaxWithLossLayer(
            class_num=self.in_channel_num,
            inputs=[{"name": "ip_last", "idxs": [0]},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))


    def _build_cifar10_model(self):
        self.attach(ConvolutionLayer([5, 5],
                                     [1, 1, 1, 1],
                                     'SAME',
                                     init_para={
                                         "name": "truncated_normal",
                                         "stddev": 1e-4},
                                     wd={"type": "l2", "scale": 0},
                                     in_channel_num=3,
                                     out_channel_num=64,
                                     name='conv1'))
        self.attach(ReLULayer(name='relu1'))
        self.attach(PoolingLayer([1, 3, 3, 1],
                                 [1, 2, 2, 1],
                                 'SAME',
                                 name='pool1'))
        self.attach(LRNLayer(name='norm1'))

        self.attach(ConvolutionLayer([5, 5],
                                     [1, 1, 1, 1],
                                     'SAME',
                                     initial_bias_value=0.1,
                                     init_para={
                                         "name": "truncated_normal",
                                         "stddev": 1e-4},
                                     wd={"type": "l2", "scale": 0},
                                     in_channel_num=64,
                                     out_channel_num=64,
                                     name='conv2'))
        self.attach(ReLULayer(name='relu2'))
        self.attach(LRNLayer(name='norm2'))
        self.attach(PoolingLayer([1, 3, 3, 1],
                                 [1, 2, 2, 1],
                                 'SAME',
                                 name='pool2'))
        self.attach(InnerProductLayer(initial_bias_value=0.1,
                                      init_para={
                                          "name": "truncated_normal",
                                          "stddev": 0.04},
                                      wd={"type": "l2", "scale": 0.004},
                                      in_channel_num=self.in_channel_num,
                                      out_channel_num=384,
                                      name='ip1'))
        self.attach(ReLULayer(name='relu3'))

        self.attach(InnerProductLayer(initial_bias_value=0.1,
                                      init_para={
                                          "name": "truncated_normal",
                                          "stddev": 0.04},
                                      wd={"type": "l2", "scale": 0.004},
                                      in_channel_num=384,
                                      out_channel_num=192,
                                      name='ip2'))
        self.attach(InnerProductLayer(initial_bias_value=0,
                                      init_para={
                                          "name": "truncated_normal",
                                          "stddev": 1/192.0},
                                      wd={"type": "l2", "scale": 0},
                                      in_channel_num=192,
                                      out_channel_num=10,
                                      name='softmax_linear'))

        self.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[{"name": "softmax_linear", "idxs": [0]},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))


class OneLayerBrain(GraphBrain):
    def __init__(self, **kwargs):
        super(OneLayerBrain, self).__init__(**kwargs)
        self.attach(
            ConvolutionLayer(ksize=[5, 5],
                             strides=[1, 1],
                             padding="SAME",
                             in_channel_num=1,
                             out_channel_num=32,
                             name="conv1")
        )
        self.attach(ReLULayer(name="relu1"))
        self.attach(
            PoolingLayer(ksize=[5, 5],
                         strides=[5, 5],
                         padding="SAME",
                         name="pool1")
        )

        self.attach(InnerProductLayer(in_channel_num=1152, out_channel_num=10, name="ip1"))
        self.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[
                {"name": "ip1", "idxs": [0]},
                {"name": "system_in", "idxs": [1]}],
            name="loss"))


class LeNet(GraphBrain):
    """
    A rough LeNet. It is supposed to copy the example from Caffe, but for the
    time being it has not been checked whether they are exactly the same.
    """
    def __init__(self, **kwargs):
        super(LeNet, self).__init__(**kwargs)
        self.attach(ConvolutionLayer(ksize=[5, 5],
                                     strides=[1, 1],
                                     padding="SAME",
                                     in_channel_num=1,
                                     out_channel_num=32,
                                     name="conv1"))
        self.attach(ReLULayer(name="relu1"))
        self.attach(PoolingLayer(ksize=[2, 2],
                                 strides=[2, 2],
                                 padding="SAME",
                                 name="pool1"))

        self.attach(ConvolutionLayer(ksize=[5, 5],
                                     strides=[1, 1],
                                     padding="SAME",
                                     in_channel_num=32,
                                     out_channel_num=64,
                                     name="conv2"))
        self.attach(ReLULayer(name="relu2"))
        self.attach(PoolingLayer(ksize=[5, 5],
                                 strides=[2, 2],
                                 padding="SAME",
                                 name="pool2"))

        self.attach(InnerProductLayer(
            in_channel_num=3136,
            out_channel_num=512,
            name="ip1"))
        self.attach(ReLULayer(name="relu3"))

        self.attach(InnerProductLayer(
            in_channel_num=512,
            out_channel_num=10,
            name="ip2"))

        self.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[{"name": "ip2", "idxs": [0]},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))


class MnistTfTutorialNet(GraphBrain):
    """
    A multiple layer network with parameters from the MNIST tutorial of
    tensorflow.
    """
    def __init__(self, **kwargs):
        super(MnistTfTutorialNet, self).__init__(**kwargs)
        self.attach(ConvolutionLayer(ksize=[5, 5],
                                     strides=[1, 1],
                                     padding="SAME",
                                     initial_bias_value=0.,
                                     init_para={"name": "truncated_normal",
                                                "stddev": 0.1},
                                     wd={"type": "l2", "scale": 5e-4},
                                     in_channel_num=1,
                                     out_channel_num=32,
                                     name="conv1"))
        self.attach(ReLULayer(name="relu1"))
        self.attach(PoolingLayer(ksize=[2, 2],
                                 strides=[2, 2],
                                 padding="SAME",
                                 name="pool1"))

        self.attach(ConvolutionLayer(ksize=[5, 5],
                                     strides=[1, 1],
                                     padding="SAME",
                                     initial_bias_value=0.1,
                                     init_para={"name": "truncated_normal",
                                                "stddev": 0.1},
                                     wd={"type": "l2", "scale": 5e-4},
                                     in_channel_num=32,
                                     out_channel_num=64,
                                     name="conv2"))
        self.attach(ReLULayer(name="relu2"))
        self.attach(PoolingLayer(ksize=[5, 5],
                                 strides=[2, 2],
                                 padding="SAME",
                                 name="pool2"))

        self.attach(InnerProductLayer(in_channel_num=3136,
                                      out_channel_num=512,
                                      initial_bias_value=0.1,
                                      init_para={"name": "truncated_normal",
                                                 "stddev": 0.1},
                                      wd={"type": "l2", "scale": 5e-4},
                                      name="ip1"))
        self.attach(ReLULayer(name="relu3"))
        self.attach(DropoutLayer(keep_prob=0.5, name="dropout1"))

        self.attach(InnerProductLayer(in_channel_num=512,
                                      out_channel_num=10,
                                      initial_bias_value=0.1,
                                      init_para={"name": "truncated_normal",
                                                 "stddev": 0.1},
                                      wd={"type": "l2", "scale": 5e-4},
                                      name="ip2"))

        self.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[{"name": "ip2", "idxs": [0]},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))


class NewMnistTfTutorialNet(GraphBrain):
    """
    A multiple layer network with parameters from the MNIST tutorial of
    tensorflow. Changed the initialization methods from truncated normal to
    default.
    """
    def __init__(self, **kwargs):
        super(NewMnistTfTutorialNet, self).__init__(**kwargs)
        self.attach(ConvolutionLayer(ksize=[5, 5],
                                     strides=[1, 1],
                                     padding="SAME",
                                     initial_bias_value=0.,
                                     init_para={"name": "default"},
                                     wd={"type": "l2", "scale": 5e-4},
                                     in_channel_num=1,
                                     out_channel_num=32,
                                     name="conv1"))
        self.attach(ReLULayer(name="relu1"))
        self.attach(PoolingLayer(ksize=[2, 2],
                                 strides=[2, 2],
                                 padding="SAME",
                                 name="pool1"))

        self.attach(ConvolutionLayer(ksize=[5, 5],
                                     strides=[1, 1],
                                     padding="SAME",
                                     initial_bias_value=0.1,
                                     init_para={"name": "default"},
                                     wd={"type": "l2", "scale": 5e-4},
                                     in_channel_num=32,
                                     out_channel_num=64,
                                     name="conv2"))
        self.attach(ReLULayer(name="relu2"))
        self.attach(PoolingLayer(ksize=[5, 5],
                                 strides=[2, 2],
                                 padding="SAME",
                                 name="pool2"))

        self.attach(InnerProductLayer(
            # The difference is caused by different way of doing padding in TF
            # and Torch.
            in_channel_num=3136 if A.backend() == A.TF else 2304,
            out_channel_num=512,
            initial_bias_value=0.1,
            init_para={"name": "default"},
            wd={"type": "l2", "scale": 5e-4},
            name="ip1"))
        self.attach(ReLULayer(name="relu3"))
        self.attach(DropoutLayer(keep_prob=0.5, name="dropout1"))

        self.attach(InnerProductLayer(in_channel_num=512,
                                      out_channel_num=10,
                                      initial_bias_value=0.1,
                                      init_para={"name": "default"},
                                      wd={"type": "l2", "scale": 5e-4},
                                      name="ip2"))

        self.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[{"name": "ip2", "idxs": [0]},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))


class VGGNet(GraphBrain):
    def __init__(self,
                 class_num=10,
                 padding="SAME",
                 loss_layer=None,
                 **kwargs):
        """
        Args:
            loss_layer: A tuple of (A python Class, A dict).
                The type of loss layer to use in this net. The first item is
                the class of the loss layer, and the second is extra parameters
                of this layer, besides `name`. If None, a softmax cross_entropy
                loss will be used.
        """
        super(VGGNet, self).__init__(**kwargs)
        self.padding = padding

        # The number counted by how many convolution layer has been applied. It
        # is used to give a easily told name to each layer.
        self.top_layer_No = 0

        self.attach_conv_bn_relu(3, 64)
        self.attach(DropoutLayer(keep_prob=0.7,
                                 name="dropout{}".format(self.top_layer_No)))
        self.attach_conv_bn_relu(64, 64)
        self.attach(PoolingLayer(ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding=self.padding,
                                 name="pool{}".format(self.top_layer_No)))

        self.attach_conv_bn_relu(64, 128)
        self.attach(DropoutLayer(keep_prob=0.6,
                                 name="dropout{}".format(self.top_layer_No)))
        self.attach_conv_bn_relu(128, 128)
        self.attach(PoolingLayer(ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding=self.padding,
                                 name="pool{}".format(self.top_layer_No)))

        self.attach_conv_bn_relu(128, 256)
        self.attach(DropoutLayer(keep_prob=0.6,
                                 name="dropout{}".format(self.top_layer_No)))
        self.attach_conv_bn_relu(256, 256)
        self.attach(DropoutLayer(keep_prob=0.6,
                                 name="dropout{}".format(self.top_layer_No)))
        self.attach_conv_bn_relu(256, 256)
        self.attach(PoolingLayer(ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding=self.padding,
                                 name="pool{}".format(self.top_layer_No)))

        self.attach_conv_bn_relu(256, 512)
        self.attach(DropoutLayer(keep_prob=0.6,
                                 name="dropout{}".format(self.top_layer_No)))
        self.attach_conv_bn_relu(512, 512)
        self.attach(DropoutLayer(keep_prob=0.6,
                                 name="dropout{}".format(self.top_layer_No)))
        self.attach_conv_bn_relu(512, 512)
        self.attach(PoolingLayer(ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding=self.padding,
                                 name="pool{}".format(self.top_layer_No)))

        self.top_layer_No += 1
        self.attach(DropoutLayer(keep_prob=0.5,
                                 name="dropout{}".format(self.top_layer_No)))
        self.attach(InnerProductLayer(
            in_channel_num=2048,
            out_channel_num=512,
            init_para={
                "name": "truncated_normal",
                "stddev": 1e-4},
            name="ip1"))
        self.attach(
            BatchNormalizationLayer(
                channel_num=512,
                name="bn{}".format(self.top_layer_No)))
        self.attach(ReLULayer(name="relu{}".format(self.top_layer_No)))

        self.top_layer_No += 1
        self.attach(DropoutLayer(keep_prob=0.5,
                                 name="dropout{}".format(self.top_layer_No)))

        self.attach(InnerProductLayer(
            in_channel_num=512,
            out_channel_num=class_num,
            init_para={
                "name": "truncated_normal",
                "stddev": 1e-4},
            name="ip2"))
        self.attach(
            BatchNormalizationLayer(
                channel_num=class_num,
                name="bn{}".format(self.top_layer_No)))

        if loss_layer:
            self.attach(loss_layer[0](
                class_num=class_num,
                name="loss",
                **loss_layer[1]))
        else:
            self.attach(SoftmaxWithLossLayer(
                class_num=class_num,
                inputs=[{"name": "ip2", "idxs": [0]},
                        {"name": "system_in", "idxs": [1]}],
                name="loss"))

    def attach_conv_bn_relu(self, in_channel_num, out_channel_num):
        """
        This method attach a block of layer, aka convolution, batch
        normalization, and ReLU, to the brain. It also maintains
        `top_layer_No`.
        """
        self.top_layer_No += 1
        self.attach(ConvolutionLayer([3, 3],
                                     [1, 1, 1, 1],
                                     padding=self.padding,
                                     init_para={
                                         "name": "truncated_normal",
                                         "stddev": 1e-4},
                                     wd={"type": "l2", "scale": 5e-4},
                                     in_channel_num=in_channel_num,
                                     out_channel_num=out_channel_num,
                                     name="conv{}".format(self.top_layer_No)))
        self.attach(BatchNormalizationLayer(
            channel_num=out_channel_num,
            name="bn{}".format(self.top_layer_No)))
        self.attach(ReLULayer(name="relu{}".format(self.top_layer_No)))


class ResNet(GraphBrain):
    def __init__(self,
                 depth=28,
                 width=2,
                 class_num=10,
                 dropout_prob=None,
                 projection_shortcut=True,
                 use_gsmax=False,
                 group_size=4,
                 **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.depth = depth
        self.width = width
        self.class_num = class_num
        self.residual_block_No = 0
        self.dropout_prob = dropout_prob
        self.projection_shortcut = projection_shortcut
        self.use_bias = None
        self.use_gsmax = use_gsmax
        self.group_size = group_size

    def _attach_stack(self,
                      n_input_plane,
                      n_output_plane,
                      count,
                      stride,
                      act_before_residual,
                      block_type="basic"):
        if block_type == "basic":
            conv_params = [[(3, 3), stride, "SAME"],
                           [(3, 3), (1, 1), "SAME"]]
        elif block_type == "bottleneck":
            # The implementation of bottleneck layer is a little tricky, though
            # I think I make it clearer than Facebook's version that uses a
            # global variable. Besides that we choose the type of residual unit
            # using the string here, in the actual `_attach_block` method, the
            # `n_input_plane` and `n_output_plane` methods stick with their
            # essential meaning when not using a bottleneck unit. That's to say
            # the real channel number when 3X3 convolution is applied. The
            # enlarging of channel number happens internally, similar with how
            # it is done in the facebook version.
            conv_params = [[(1, 1), (1, 1), "SAME"],
                           [(3, 3), stride, "SAME"],
                           [(1, 1), (1, 1), "SAME"]]
        else:
            raise Exception("Block type {} is not supported.".format(block_type))

        self._attach_block(n_input_plane,
                           n_output_plane,
                           stride,
                           act_before_residual,
                           conv_params)
        for i in range(2, count+1):
            if block_type == "basic":
                conv_params = [[(3, 3), (1, 1), "SAME"],
                               [(3, 3), (1, 1), "SAME"]]
            elif block_type == "bottleneck":
                conv_params = [[(1, 1), (1, 1), "SAME"],
                               [(3, 3), (1, 1), "SAME"],
                               [(1, 1), (1, 1), "SAME"]]
            else:
                raise Exception("Block type {} is not supported.".format(block_type))

            self._attach_block(n_output_plane,
                               n_output_plane,
                               (1, 1),
                               False,
                               conv_params)

    def _attach_block(self,
                      n_input_plane,
                      n_output_plane,
                      stride,
                      act_before_residual,
                      conv_params):
        self.residual_block_No += 1

        main_branch_layer_name = self.blocks[-1].name

        is_bottleneck = False  # A flag for shortcut padding.
        for i, v in enumerate(conv_params):
            ksize, r_stride, padding = v

            in_channel_num = n_output_plane
            if i == 0:
                in_channel_num = n_input_plane

            self.attach(BatchNormalizationLayer(
                channel_num=in_channel_num,
                dim_num=2,
                name="bn_{}_{}".format(self.residual_block_No, i)))

            if self.use_gsmax and n_input_plane > 16:
                self.attach(GroupSoftmaxLayer(
                    group_size=self.group_size*n_input_plane/(16*self.width),
                    name="gsmax_{}_{}".format(self.residual_block_No, i)))
            else:
                self.attach(ReLULayer(name="relu_{}_{}".format(
                    self.residual_block_No, i)))

            if i == 0:
                if n_input_plane != n_output_plane and act_before_residual:
                    # At the first block of each BIG layer, two branches may
                    # share the same BN and activation function, thus bookkeep
                    # the branching layer name.
                    main_branch_layer_name = self.blocks[-1].name

            # We determine whether this is bottleneck layer by checking the
            # kernel size.
            if i == len(conv_params) - 1 and list(ksize) == [1, 1]:
                # This is the last layer of a bottleneck layer, we need to
                # increase the channel number back.
                is_bottleneck = True
                # In wide residual network, only bottleneck conv layer is
                # widened, thus the effect needs be offset back.
                out_channel_num = n_output_plane * 4 / self.width
            else:
                out_channel_num = n_output_plane

            if i != 0:
                if self.dropout_prob:
                    self.attach(DropoutLayer(
                        keep_prob=1-self.dropout_prob,
                        name="dropout_{}_{}".format(self.residual_block_No,
                                                    i)))
            self.attach(ConvolutionLayer(ksize,
                                         [1, r_stride[0], r_stride[1], 1],
                                         padding=padding,
                                         init_para={"name": "msra"},
                                         initial_bias_value=self.use_bias,
                                         wd=self.wd,
                                         in_channel_num=in_channel_num,
                                         out_channel_num=out_channel_num,
                                         name="conv_{}_{}".format(
                                             self.residual_block_No, i)))

        last_residual_layer_name = self.blocks[-1].name

        if n_input_plane != n_output_plane:
            if self.projection_shortcut:
                self.attach(ConvolutionLayer(
                    [1, 1],
                    [1, stride[0], stride[1], 1],
                    inputs=[{"name": main_branch_layer_name}],
                    padding="SAME",
                    init_para={"name": "msra"},
                    initial_bias_value=self.use_bias,
                    wd=self.wd,
                    in_channel_num=n_input_plane,
                    out_channel_num=out_channel_num,
                    name="conv_{}_shortcut".format(self.residual_block_No)))
            else:
                _ = (1, stride[0], stride[1], 1)
                if is_bottleneck:
                    in_channel_num = n_input_plane * 4
                else:
                    in_channel_num = n_input_plane

                self.attach(PoolingLayer(
                    ksize=_,
                    strides=_,
                    inputs=[{"name": main_branch_layer_name}],
                    padding="VALID",
                    type="avg",
                    name="pool_{}_shortcut".format(self.residual_block_No)))
                if A.DATA_FORMAT == "CHW":
                    padding = [0,(out_channel_num - in_channel_num) // 2, 0, 0]
                else:
                    padding = [0, 0, 0, (out_channel_num - in_channel_num) // 2]
                self.attach(PaddingLayer(
                    padding=padding,
                    name="pad_{}_shortcut".format(self.residual_block_No)))

            shortcut_layer_name = self.blocks[-1].name
        else:
            shortcut_layer_name = main_branch_layer_name

        self.attach(MergeLayer(inputs=[{"name": last_residual_layer_name},
                                       {"name": shortcut_layer_name}],
                               name="merge_{}".format(self.residual_block_No)))


class CifarResNet(ResNet):
    def __init__(self,
                 color_channel_num=3,
                 pool_size=8,
                 n_stages = None,
                 sub_class_multiplier_ratio=0.5,
                 h_loss=False,
                 **kwargs):
        super(CifarResNet, self).__init__(**kwargs)

        depth = self.depth

        assert((depth - 4) % 6 == 0), \
            "Depth must be 6 * n + 4."
        k = self.width
        if not n_stages:
            n_stages = [16, 16*k, 32*k, 64*k]
        assert (depth - 4) % 6 is 0
        n = (depth - 4) // 6
        if self.projection_shortcut:
            act_before_residual = [True, True, True]
            self.wd = {"type": "l2", "scale": 5e-4}
        else:
            act_before_residual = [True, False, False]
            self.wd = {"type": "l2", "scale": 0.0002}

        self.attach(ConvolutionLayer([3, 3],
                                     [1, 1, 1, 1],
                                     padding="SAME",
                                     init_para={"name": "msra"},
                                     wd=self.wd,
                                     in_channel_num=color_channel_num,
                                     out_channel_num=n_stages[0],
                                     initial_bias_value=self.use_bias,
                                     name="conv0"))

        self._attach_stack(n_input_plane=n_stages[0],
                           n_output_plane=n_stages[1],
                           count=n,
                           stride=(1, 1),
                           act_before_residual=act_before_residual[0])
        self._attach_stack(n_input_plane=n_stages[1],
                           n_output_plane=n_stages[2],
                           count=n,
                           stride=(2, 2),
                           act_before_residual=act_before_residual[1])
        self._attach_stack(n_input_plane=n_stages[2],
                           n_output_plane=n_stages[3],
                           count=n,
                           stride=(2, 2),
                           act_before_residual=act_before_residual[2])
        self.attach(BatchNormalizationLayer(n_stages[3], dim_num=2, name="bn_out"))
        if self.use_gsmax:
            self.attach(GroupSoftmaxLayer(
                group_size=self.group_size*(n_stages[3], n_stages[1]),
                name="gsmax_out"))
        else:
            self.attach(ReLULayer(name="relu_out"))
        self.attach(PoolingLayer(ksize=[1, pool_size, pool_size, 1],
                                 strides=[1, 1, 1, 1],
                                 padding="VALID",
                                 type="avg",
                                 name="global_pool"))
        self.attach(ReshapeLayer(name="reshape"))
        self.attach(InnerProductLayer(initial_bias_value=0,
                                      init_para={"name": "default"},
                                      wd=self.wd,
                                      in_channel_num=n_stages[3],
                                      out_channel_num=self.class_num,
                                      name='ip'))
        if h_loss:
            self.attach(SoftmaxWithLossLayer(
                class_num=self.class_num,
                multiplier=sub_class_multiplier_ratio,
                inputs=[{"name": "ip"},
                        {"name": "system_in", "idxs": [2]}],
                name="softmax"))
            self.attach(CollapseOutLayer(group_size=5,
                                         type="average_out",
                                         inputs=[
                                             {"name": "ip"}
                                         ],
                                         name="average_out"))
            self.attach(SoftmaxWithLossLayer(
                class_num=20,
                multiplier=1-sub_class_multiplier_ratio,
                inputs=[
                    {"name": "average_out"},
                    {"name": "system_in", "idxs": [1]}],
                name="super_class_loss"))
        else:
            self.attach(SoftmaxWithLossLayer(
                class_num=self.class_num,
                inputs=[{"name": "ip"},
                        {"name": "system_in", "idxs": [1]}],
                name="softmax"))


class ImagenetResNet(ResNet):
    def __init__(self, **kwargs):
        super(ImagenetResNet, self).__init__(**kwargs)

        k = self.width
        n_stages = [64, 64*k, 128*k, 256*k, 512*k]
        # Given imagenet needs a some non-regular conv-bn-relu-pool block at
        # the beginning of the branch, it is tailed by handle, and replaces the
        # first shared pre-activation block. The remaining shared
        # pre-activation block are still active.
        act_before_residual = [False, True, True, True]
        strides = [1, 2, 2, 2]
        # The configuration to train imagenet.
        cfg = {
         18  : [[2, 2, 2, 2], "basicblock"],
         34  : [[3, 4, 6, 3], "basicblock"],
         50  : [[3, 4, 6, 3], "bottleneck"],
         101 : [[3, 4, 23, 3], "bottleneck"],
         152 : [[3, 8, 36, 3], "bottleneck"],
         200 : [[3, 24, 36, 3], "bottleneck"],
        }
        n_depth, block_name = cfg[self.depth]
        self.wd = {"type": "l2", "scale": 1e-4}

        self.attach(ConvolutionLayer([7, 7],
                                     [1, 2, 2, 1],
                                     padding="SAME",
                                     init_para={"name": "msra"},
                                     wd=self.wd,
                                     out_channel_num=64,
                                     initial_bias_value=self.use_bias,
                                     name="conv0"))
        self.attach(BatchNormalizationLayer(dim_num=2, name="bn0"))
        self.attach(ReLULayer(name="relu0"))
        self.attach(PoolingLayer(ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1],
                                 padding="SAME",
                                 type="max",
                                 name="pool0"))
        for i in range(4):
            self._attach_stack(n_input_plane=n_stages[i],
                               n_output_plane=n_stages[i+1],
                               count=n_depth[i],
                               stride=(strides[i], strides[i]),
                               act_before_residual=act_before_residual[i],
                               block_type="bottleneck")

        self.attach(ReLULayer(name="relu_final"))
        self.attach(PoolingLayer(ksize=[1, 7, 7, 1],
                                 strides=[1, 1, 1, 1],
                                 padding="VALID",
                                 type="avg",
                                 name="global_pool"))
        self.attach(ReshapeLayer(name="reshape"))
        self.attach(InnerProductLayer(initial_bias_value=0,
                                      init_para={"name": "default"},
                                      wd=self.wd,
                                      out_channel_num=self.class_num,
                                      name='ip'))
        self.attach(SoftmaxWithLossLayer(
            class_num=self.class_num,
            inputs=[{"name": "ip"},
                    {"name": "system_in", "idxs": [1]}],
            name="softmax"))


class PatriceNet(GraphBrain):
    """
    Reproduction of the paper: *Best Practices for Convolutional Neural
    Networks Applied to Visual Document Analysis*.
    """
    def __init__(self, **kwargs):
        super(PatriceNet, self).__init__(**kwargs)

        self.attach(InnerProductLayer(in_channel_num=784,
                                      out_channel_num=800,
                                      # init_para={"name": "normal",
                                      #            "stddev": 0.05},
                                      initial_bias_value=0,
                                      wd=None,
                                      name="ip1"))
        self.attach(ReLULayer(name="relu1"))
        self.attach(InnerProductLayer(in_channel_num=800,
                                      out_channel_num=10,
                                      initial_bias_value=0,
                                      # init_para={"name": "normal",
                                      #            "stddev": 0.05},
                                      wd=None,
                                      name="ip2"))
        self.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[
                {"name": "ip2", "idxs": [0]},
                {"name": "system_in", "idxs": [1]}],
            name="loss"))


class IntriguingConvNet(GraphBrain):
    """
    The modified version of simple convnet on MNIST from *Intriguing properties
    of neural networks*.
    """
    def __init__(self, **kwargs):
        super(IntriguingConvNet, self).__init__(**kwargs)

        self.attach(ConvolutionLayer(ksize=[5, 5],
                                     strides=[1, 1],
                                     padding="SAME",
                                     in_channel_num=1,
                                     out_channel_num=32,
                                     name="conv1"))
        self.attach(ReLULayer(name="relu1"))
        self.attach(ConvolutionLayer(ksize=[5, 5],
                                     strides=[2, 2],
                                     padding="SAME",
                                     in_channel_num=32,
                                     out_channel_num=64,
                                     name="conv2"))
        self.attach(ReLULayer(name="relu2"))
        self.attach(InnerProductLayer(in_channel_num=10816,
                                      out_channel_num=64,
                                      initial_bias_value=0,
                                      # init_para={"name": "normal",
                                      #            "stddev": 0.05},
                                      # wd=None,
                                      name="ip1"))
        self.attach(ReLULayer(name="relu3",
                              summarize_output=True,
        ))
        self.attach(InnerProductLayer(in_channel_num=64,
                                      out_channel_num=10,
                                      initial_bias_value=0,
                                      # init_para={"name": "normal",
                                      #            "stddev": 0.05},
                                      wd=None,
                                      name="ip2"))
        self.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[
                {"name": "ip2", "idxs": [0]},
                {"name": "system_in", "idxs": [1]}],
            name="loss"))


class AllConvNet(GraphBrain):
    """
    All Convolutional Network on CIFAR-10.
    """
    def __init__(self, **kwargs):
        super(AllConvNet, self).__init__(**kwargs)

        self.attach(DropoutLayer(keep_prob=0.8, name="dropout1"))
        self.attach(ConvolutionLayer(ksize=[3, 3],
                                     strides=[1, 1],
                                     padding="SAME",
                                     init_para={"name": "xavier"},
                                     wd={"type": "l2", "scale": 0.001},
                                     in_channel_num=3,
                                     out_channel_num=96,
                                     name="conv1"))
        self.attach(ReLULayer(name="relu1"))
        self.attach(ConvolutionLayer(ksize=[3, 3],
                                     strides=[1, 1],
                                     padding="SAME",
                                     init_para={"name": "xavier"},
                                     wd={"type": "l2", "scale": 0.001},
                                     in_channel_num=96,
                                     out_channel_num=96,
                                     name="conv2"))
        self.attach(ReLULayer(name="relu2"))
        self.attach(ConvolutionLayer(ksize=[3, 3],
                                     strides=[2, 2],
                                     padding="SAME",
                                     init_para={"name": "xavier"},
                                     wd={"type": "l2", "scale": 0.001},
                                     in_channel_num=96,
                                     out_channel_num=96,
                                     name="conv3"))
        self.attach(DropoutLayer(keep_prob=0.5, name="dropout2"))
        self.attach(ReLULayer(name="relu3"))
        self.attach(ConvolutionLayer(ksize=[3, 3],
                                     strides=[1, 1],
                                     padding="SAME",
                                     init_para={"name": "xavier"},
                                     wd={"type": "l2", "scale": 0.001},
                                     in_channel_num=96,
                                     out_channel_num=192,
                                     name="conv4"))
        self.attach(ReLULayer(name="relu4"))
        self.attach(ConvolutionLayer(ksize=[3, 3],
                                     strides=[1, 1],
                                     padding="SAME",
                                     init_para={"name": "xavier"},
                                     wd={"type": "l2", "scale": 0.001},
                                     in_channel_num=192,
                                     out_channel_num=192,
                                     name="conv5"))
        self.attach(ReLULayer(name="relu5"))
        self.attach(ConvolutionLayer(ksize=[3, 3],
                                     strides=[2, 2],
                                     padding="SAME",
                                     init_para={"name": "xavier"},
                                     wd={"type": "l2", "scale": 0.001},
                                     in_channel_num=192,
                                     out_channel_num=192,
                                     name="conv6"))
        self.attach(DropoutLayer(keep_prob=0.5, name="dropout3"))
        self.attach(ReLULayer(name="relu6"))
        self.attach(ConvolutionLayer(ksize=[3, 3],
                                     strides=[1, 1],
                                     padding="VALID",
                                     init_para={"name": "xavier"},
                                     wd={"type": "l2", "scale": 0.001},
                                     in_channel_num=192,
                                     out_channel_num=192,
                                     name="conv7"))
        self.attach(ReLULayer(name="relu7"))
        self.attach(ConvolutionLayer(ksize=[1, 1],
                                     strides=[1, 1],
                                     padding="VALID",
                                     init_para={"name": "xavier"},
                                     wd={"type": "l2", "scale": 0.001},
                                     in_channel_num=192,
                                     out_channel_num=192,
                                     name="conv8"))
        self.attach(ReLULayer(name="relu8"))
        self.attach(ConvolutionLayer(ksize=[1, 1],
                                     strides=[1, 1],
                                     padding="VALID",
                                     init_para={"name": "xavier"},
                                     wd={"type": "l2", "scale": 0.001},
                                     in_channel_num=192,
                                     out_channel_num=10,
                                     name="conv9"))
        self.attach(ReLULayer(name="relu9"))
        self.attach(PoolingLayer(ksize=[6, 6],
                                 strides=[1, 1],
                                 padding="VALID",
                                 type="avg",
                                 name="global_pool"))
        self.attach(ReshapeLayer(name="reshape"))
        self.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[
                {"name": "reshape", "idxs": [0]},
                {"name": "system_in", "idxs": [1]}],
            name="loss"))

class AllConvImagenet(GraphBrain):
    """
    All convolutional style AlexNet on ImageNet.
    """
    def __init__(self, **kwargs):
        super(AllConvImagenet, self).__init__(**kwargs)
        wd = {"type": "l2", "scale": 1 * 10e-4}
        # wd = None
        # init = {"name": "msra"}
        init = {"name": "xavier"}
        self.attach(ConvolutionLayer(ksize=[11, 11],
                                     strides=[4, 4],
                                     padding="VALID",
                                     init_para=init,
                                     wd=wd,
                                     in_channel_num=3,
                                     out_channel_num=96,
                                     name="conv1"))
        self.attach(ReLULayer(name="relu1"))
        self.attach(ConvolutionLayer(ksize=[1, 1],
                                     strides=[1, 1],
                                     padding="VALID",
                                     init_para=init,
                                     wd=wd,
                                     in_channel_num=96,
                                     out_channel_num=96,
                                     name="conv2"))
        self.attach(ReLULayer(name="relu2"))
        self.attach(ConvolutionLayer(ksize=[3, 3],
                                     strides=[2, 2],
                                     padding="SAME",
                                     init_para=init,
                                     wd=wd,
                                     in_channel_num=96,
                                     out_channel_num=96,
                                     name="conv3"))
        self.attach(ReLULayer(name="relu3"))
        self.attach(ConvolutionLayer(ksize=[5, 5],
                                     strides=[1, 1],
                                     padding="SAME",
                                     init_para=init,
                                     wd=wd,
                                     in_channel_num=96,
                                     out_channel_num=256,
                                     name="conv4"))
        self.attach(ReLULayer(name="relu4"))
        self.attach(ConvolutionLayer(ksize=[1, 1],
                                     strides=[1, 1],
                                     padding="SAME",
                                     init_para=init,
                                     wd=wd,
                                     in_channel_num=256,
                                     out_channel_num=256,
                                     name="conv5"))
        self.attach(ReLULayer(name="relu5"))
        self.attach(ConvolutionLayer(ksize=[3, 3],
                                     strides=[2, 2],
                                     padding="SAME",
                                     init_para=init,
                                     wd=wd,
                                     in_channel_num=256,
                                     out_channel_num=256,
                                     name="conv6"))
        self.attach(ReLULayer(name="relu6"))
        self.attach(ConvolutionLayer(ksize=[3, 3],
                                     strides=[1, 1],
                                     padding="SAME",
                                     init_para=init,
                                     wd=wd,
                                     in_channel_num=256,
                                     out_channel_num=384,
                                     name="conv7"))
        self.attach(ReLULayer(name="relu7"))
        self.attach(ConvolutionLayer(ksize=[1, 1],
                                     strides=[1, 1],
                                     padding="SAME",
                                     init_para=init,
                                     wd=wd,
                                     in_channel_num=384,
                                     out_channel_num=384,
                                     name="conv8"))
        self.attach(ReLULayer(name="relu8"))
        self.attach(ConvolutionLayer(ksize=[3, 3],
                                     strides=[2, 2],
                                     padding="SAME",
                                     init_para=init,
                                     wd=wd,
                                     in_channel_num=384,
                                     out_channel_num=384,
                                     name="conv9"))
        self.attach(DropoutLayer(keep_prob=0.5, name="dropout1"))
        self.attach(ReLULayer(name="relu9"))
        self.attach(ConvolutionLayer(ksize=[3, 3],
                                     strides=[1, 1],
                                     padding="SAME",
                                     init_para=init,
                                     wd=wd,
                                     in_channel_num=384,
                                     out_channel_num=1024,
                                     name="conv10"))
        self.attach(ReLULayer(name="relu10"))
        self.attach(ConvolutionLayer(ksize=[1, 1],
                                     strides=[1, 1],
                                     padding="SAME",
                                     init_para=init,
                                     wd=wd,
                                     in_channel_num=1024,
                                     out_channel_num=1024,
                                     name="conv11"))
        self.attach(ReLULayer(name="relu11"))
        self.attach(ConvolutionLayer(ksize=[1, 1],
                                     strides=[1, 1],
                                     padding="SAME",
                                     init_para=init,
                                     wd=wd,
                                     in_channel_num=1024,
                                     out_channel_num=1000,
                                     name="conv12"))
        self.attach(ReLULayer(name="relu12"))
        self.attach(PoolingLayer(ksize=[5, 5],
                                 strides=[1, 1],
                                 padding="VALID",
                                 type="avg",
                                 name="global_pool"))
        self.attach(ReshapeLayer(name="reshape"))
        self.attach(SoftmaxWithLossLayer(
            class_num=1000,
            inputs=[
                {"name": "reshape", "idxs": [0]},
                {"name": "system_in", "idxs": [1]}],
            name="loss"))
