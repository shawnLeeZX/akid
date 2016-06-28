from akid import Brain
from akid.layers import (
    ConvolutionLayer,
    PoolingLayer,
    ReLULayer,
    InnerProductLayer,
    SoftmaxWithLossLayer,
    LRNLayer,
    BatchNormalizationLayer,
    DropoutLayer
)


class AlexNet(Brain):
    """
    A class for alex net specifically.
    """
    def __init__(self, **kwargs):
        super(AlexNet, self).__init__(**kwargs)

        self.attach(ConvolutionLayer([5, 5],
                                     [1, 1, 1, 1],
                                     'SAME',
                                     init_para={
                                         "name": "truncated_normal",
                                         "stddev": 1e-4},
                                     wd={"type": "l2", "scale": 0},
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
                                      out_channel_num=384,
                                      name='ip1'))
        self.attach(ReLULayer(name='relu3'))

        self.attach(InnerProductLayer(initial_bias_value=0.1,
                                      init_para={
                                          "name": "truncated_normal",
                                          "stddev": 0.04},
                                      wd={"type": "l2", "scale": 0.004},
                                      out_channel_num=192,
                                      name='ip2'))
        self.attach(InnerProductLayer(initial_bias_value=0,
                                      init_para={
                                          "name": "truncated_normal",
                                          "stddev": 1/192.0},
                                      wd={"type": "l2", "scale": 0},
                                      out_channel_num=10,
                                      name='softmax_linear'))

        self.attach(SoftmaxWithLossLayer(class_num=10, name='loss'))


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
        self.attach(SoftmaxWithLossLayer(class_num=10, name="loss"))


class LeNet(Brain):
    """
    A rough LeNet. It is supposed to copy the example from Caffe, but for the
    time being it has not been checked whether they are exactly the same.
    """
    def __init__(self, **kwargs):
        super(LeNet, self).__init__(**kwargs)
        self.attach(ConvolutionLayer(ksize=[5, 5],
                                     strides=[1, 1, 1, 1],
                                     padding="SAME",
                                     out_channel_num=32,
                                     name="conv1"))
        self.attach(ReLULayer(name="relu1"))
        self.attach(PoolingLayer(ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding="SAME",
                                 name="pool1"))

        self.attach(ConvolutionLayer(ksize=[5, 5],
                                     strides=[1, 1, 1, 1],
                                     padding="SAME",
                                     out_channel_num=64,
                                     name="conv2"))
        self.attach(ReLULayer(name="relu2"))
        self.attach(PoolingLayer(ksize=[1, 5, 5, 1],
                                 strides=[1, 2, 2, 1],
                                 padding="SAME",
                                 name="pool2"))

        self.attach(InnerProductLayer(out_channel_num=512, name="ip1"))
        self.attach(ReLULayer(name="relu3"))

        self.attach(InnerProductLayer(out_channel_num=10, name="ip2"))

        self.attach(SoftmaxWithLossLayer(class_num=10, name="loss"))


class MnistTfTutorialNet(Brain):
    """
    A multiple layer network with parameters from the MNIST tutorial of
    tensorflow.
    """
    def __init__(self, **kwargs):
        super(MnistTfTutorialNet, self).__init__(**kwargs)
        self.attach(ConvolutionLayer(ksize=[5, 5],
                                     strides=[1, 1, 1, 1],
                                     padding="SAME",
                                     initial_bias_value=0.,
                                     init_para={"name": "truncated_normal",
                                                "stddev": 0.1},
                                     wd={"type": "l2", "scale": 5e-4},
                                     out_channel_num=32,
                                     name="conv1"))
        self.attach(ReLULayer(name="relu1"))
        self.attach(PoolingLayer(ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding="SAME",
                                 name="pool1"))

        self.attach(ConvolutionLayer(ksize=[5, 5],
                                     strides=[1, 1, 1, 1],
                                     padding="SAME",
                                     initial_bias_value=0.1,
                                     init_para={"name": "truncated_normal",
                                                "stddev": 0.1},
                                     wd={"type": "l2", "scale": 5e-4},
                                     out_channel_num=64,
                                     name="conv2"))
        self.attach(ReLULayer(name="relu2"))
        self.attach(PoolingLayer(ksize=[1, 5, 5, 1],
                                 strides=[1, 2, 2, 1],
                                 padding="SAME",
                                 name="pool2"))

        self.attach(InnerProductLayer(out_channel_num=512,
                                      initial_bias_value=0.1,
                                      init_para={"name": "truncated_normal",
                                                 "stddev": 0.1},
                                      wd={"type": "l2", "scale": 5e-4},
                                      name="ip1"))
        self.attach(ReLULayer(name="relu3"))
        self.attach(DropoutLayer(keep_prob=0.5, name="dropout1"))

        self.attach(InnerProductLayer(out_channel_num=10,
                                      initial_bias_value=0.1,
                                      init_para={"name": "truncated_normal",
                                                 "stddev": 0.1},
                                      wd={"type": "l2", "scale": 5e-4},
                                      name="ip2"))

        self.attach(SoftmaxWithLossLayer(class_num=10, name="loss"))


class VGGNet(Brain):
    def __init__(self, class_num=10, padding="SAME", **kwargs):
        super(VGGNet, self).__init__(**kwargs)
        self.padding = padding

        # The number counted by how many convolution layer has been applied. It
        # is used to give a easily told name to each layer.
        self.top_layer_No = 0

        self.attach_conv_bn_relu(64)
        self.attach(DropoutLayer(keep_prob=0.7,
                                 name="dropout{}".format(self.top_layer_No)))
        self.attach_conv_bn_relu(64)
        self.attach(PoolingLayer(ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding=self.padding,
                                 name="pool{}".format(self.top_layer_No)))

        self.attach_conv_bn_relu(128)
        self.attach(DropoutLayer(keep_prob=0.6,
                                 name="dropout{}".format(self.top_layer_No)))
        self.attach_conv_bn_relu(128)
        self.attach(PoolingLayer(ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding=self.padding,
                                 name="pool{}".format(self.top_layer_No)))

        self.attach_conv_bn_relu(256)
        self.attach(DropoutLayer(keep_prob=0.6,
                                 name="dropout{}".format(self.top_layer_No)))
        self.attach_conv_bn_relu(256)
        self.attach(DropoutLayer(keep_prob=0.6,
                                 name="dropout{}".format(self.top_layer_No)))
        self.attach_conv_bn_relu(256)
        self.attach(PoolingLayer(ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding=self.padding,
                                 name="pool{}".format(self.top_layer_No)))

        self.attach_conv_bn_relu(512)
        self.attach(DropoutLayer(keep_prob=0.6,
                                 name="dropout{}".format(self.top_layer_No)))
        self.attach_conv_bn_relu(512)
        self.attach(DropoutLayer(keep_prob=0.6,
                                 name="dropout{}".format(self.top_layer_No)))
        self.attach_conv_bn_relu(512)
        self.attach(PoolingLayer(ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding=self.padding,
                                 name="pool{}".format(self.top_layer_No)))

        self.top_layer_No += 1
        self.attach(DropoutLayer(keep_prob=0.5,
                                 name="dropout{}".format(self.top_layer_No)))
        self.attach(InnerProductLayer(out_channel_num=512,
                                      init_para={
                                          "name": "truncated_normal",
                                          "stddev": 1e-4},
                                      name="ip1"))
        self.attach(
            BatchNormalizationLayer(name="bn{}".format(self.top_layer_No)))
        self.attach(ReLULayer(name="relu{}".format(self.top_layer_No)))

        self.top_layer_No += 1
        self.attach(DropoutLayer(keep_prob=0.5,
                                 name="dropout{}".format(self.top_layer_No)))
        self.attach(InnerProductLayer(out_channel_num=class_num,
                                      init_para={
                                          "name": "truncated_normal",
                                          "stddev": 1e-4},
                                      name="ip2"))
        self.attach(SoftmaxWithLossLayer(class_num=class_num, name="loss"))

    def attach_conv_bn_relu(self, out_channel_num):
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
                                     out_channel_num=out_channel_num,
                                     name="conv{}".format(self.top_layer_No)))
        self.attach(BatchNormalizationLayer(
            name="bn{}".format(self.top_layer_No)))
        self.attach(ReLULayer(name="relu{}".format(self.top_layer_No)))
