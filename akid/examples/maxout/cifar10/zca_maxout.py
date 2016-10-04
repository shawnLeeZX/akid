from akid import AKID_DATA_PATH
from akid.core import kids, kongfus
from akid import Cifar10FeedSource, FeedSensor, Brain
from akid.layers import (
    ConvolutionLayer,
    PoolingLayer,
    InnerProductLayer,
    SoftmaxWithLossLayer,
    DropoutLayer,
    MaxoutLayer
)


def setup():
    # Set up brain
    # #########################################################################
    brain = Brain(name='maxout-zca-cifar10')

    brain.attach(DropoutLayer(keep_prob=0.8, name='dropout1'))

    brain.attach(ConvolutionLayer([8, 8],
                                  [1, 1, 1, 1],
                                  'SAME',
                                  init_para={
                                      "name": "uniform",
                                      "range": 0.005},
                                  wd={"type": "l2", "scale": 0.0005},
                                  out_channel_num=192,
                                  name='conv1'))
    brain.attach(PoolingLayer([1, 4, 4, 1],
                              [1, 2, 2, 1],
                              'SAME',
                              name='pool1'))
    brain.attach(MaxoutLayer(name='maxout1'))
    brain.attach(DropoutLayer(keep_prob=0.5, name='dropout2'))

    brain.attach(ConvolutionLayer([8, 8],
                                  [1, 1, 1, 1],
                                  'SAME',
                                  init_para={
                                      "name": "uniform",
                                      "range": 0.005},
                                  wd={"type": "l2", "scale": 0.0005},
                                  out_channel_num=384,
                                  name='conv2'))
    brain.attach(PoolingLayer([1, 4, 4, 1],
                              [1, 2, 2, 1],
                              'SAME',
                              name='pool2'))
    brain.attach(MaxoutLayer(name='maxout2'))
    brain.attach(DropoutLayer(keep_prob=0.5, name='dropout3'))

    brain.attach(ConvolutionLayer([5, 5],
                                  [1, 1, 1, 1],
                                  'SAME',
                                  init_para={
                                      "name": "uniform",
                                      "range": 0.005},
                                  wd={"type": "l2", "scale": 0.0005},
                                  out_channel_num=384,
                                  name='conv3'))
    brain.attach(PoolingLayer([1, 2, 2, 1],
                              [1, 2, 2, 1],
                              'SAME',
                              name='pool3'))
    brain.attach(MaxoutLayer(name='maxout3'))
    brain.attach(DropoutLayer(keep_prob=0.5, name='dropout3'))

    brain.attach(InnerProductLayer(init_para={"name": "uniform",
                                              "range": 0.005},
                                   wd={"type": "l2", "scale": 0.004},
                                   out_channel_num=2500,
                                   name='ip1'))
    brain.attach(MaxoutLayer(group_size=5, name='maxout4'))
    brain.attach(DropoutLayer(keep_prob=0.5, name='dropout3'))

    brain.attach(InnerProductLayer(init_para={"name": "uniform",
                                              "range": 0.005},
                                   wd={"type": "l2", "scale": 0},
                                   out_channel_num=10,
                                   name='softmax_linear'))

    brain.attach(SoftmaxWithLossLayer(class_num=10, name='loss'))

    # Set up a sensor.
    # #########################################################################
    cifar_source = Cifar10FeedSource(
        name="CIFAR10",
        url='http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
        work_dir=AKID_DATA_PATH + '/cifar10',
        use_zca=True,
        num_train=50000,
        num_val=10000)

    sensor = FeedSensor(source_in=cifar_source,
                        batch_size=128,
                        name='data')

    # Summon a survivor.
    # #########################################################################
    survivor = kids.Kid(
        sensor,
        brain,
        kongfus.MomentumKongFu(base_lr=0.025,
                               momentum=0.5,
                               decay_rate=0.1,
                               decay_epoch_num=50),
        max_steps=200000)

    survivor.setup()
    return survivor

if __name__ == "__main__":
    # Start training
    # #######################################################################
    kid = setup()
    kid.practice()
