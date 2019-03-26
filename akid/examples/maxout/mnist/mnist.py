from __future__ import absolute_import
from akid.sugar.block_templates import cnn_block
from akid import AKID_DATA_PATH
from akid import Brain, MNISTFeedSource, FeedSensor, Kid, MomentumKongFu
from akid.layers import DropoutLayer


def setup():
    brain = Brain(name="maxout_mnist")

    bn = {"gamma_init": 1, "fix_gamma": True}

    brain.attach(DropoutLayer(keep_prob=0.8, name='dropout0'))

    brain.attach(cnn_block(ksize=[8, 8],
                           initial_bias_value=0.,
                           init_para={"name": "uniform",
                                      "range": 0.005},
                           wd={"type": "l2", "scale": 5e-4},
                           out_channel_num=48*2,
                           pool_size=[4, 4],
                           pool_stride=[2, 2],
                           activation={"type": "maxout", "group_size": 2},
                           keep_prob=0.5,
                           bn=bn))

    brain.attach(cnn_block(ksize=[8, 8],
                           initial_bias_value=0.,
                           init_para={"name": "uniform",
                                      "range": 0.005},
                           wd={"type": "l2", "scale": 5e-4},
                           out_channel_num=48*2,
                           pool_size=[4, 4],
                           pool_stride=[2, 2],
                           activation={"type": "maxout", "group_size": 2},
                           keep_prob=0.5,
                           bn=bn))

    brain.attach(cnn_block(ksize=[5, 5],
                           initial_bias_value=0,
                           init_para={"name": "uniform",
                                      "range": 0.005},
                           wd={"type": "l2", "scale": 5e-4},
                           out_channel_num=24*4,
                           pool_size=[2, 2],
                           pool_stride=[2, 2],
                           activation={"type": "maxout", "group_size": 4},
                           bn=bn,
                           keep_prob=0.5))

    brain.attach(cnn_block(ksize=None,
                           initial_bias_value=0,
                           init_para={"name": "uniform",
                                      "range": 0.005},
                           wd={"type": "l2", "scale": 5e-4},
                           out_channel_num=10,
                           bn=bn,
                           activation={"type": "softmax"}))

    source = MNISTFeedSource(name="MNIST",
                             url='http://yann.lecun.com/exdb/mnist/',
                             work_dir=AKID_DATA_PATH + '/mnist',
                             num_train=60000,
                             num_val=10000,
                             center=True,
                             scale=True)

    kid = Kid(FeedSensor(name='data',
                         source_in=source,
                         batch_size=128,
                         val_batch_size=100),
              brain,
              MomentumKongFu(momentum=0.9,
                             base_lr=1,
                             decay_rate=0.95,
                             decay_epoch_num=1),
              max_steps=20000)
    kid.setup()

    return kid


if __name__ == "__main__":
    kid = setup()
    kid.practice()
