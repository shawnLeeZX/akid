from __future__ import absolute_import
from akid.sugar.block_templates import cnn_block
from akid import AKID_DATA_PATH
from akid import GraphBrain, MNISTFeedSource, FeedSensor, Kid, MomentumKongFu
from akid.layers import SoftmaxWithLossLayer
from akid import LearningRateScheme


def setup(bn=None, activation_before_pooling=False):
    brain = GraphBrain(name="sugar_mnist")

    brain.attach(cnn_block(ksize=[5, 5],
                           initial_bias_value=0.,
                           init_para={"name": "truncated_normal",
                                      "stddev": 0.1},
                           wd={"type": "l2", "scale": 5e-4},
                           in_channel_num=1,
                           out_channel_num=32,
                           pool_size=[2, 2],
                           pool_stride=[2, 2],
                           activation={"type": "relu"},
                           activation_before_pooling=activation_before_pooling,
                           bn=bn))

    brain.attach(cnn_block(ksize=[5, 5],
                           initial_bias_value=0.,
                           init_para={"name": "truncated_normal",
                                      "stddev": 0.1},
                           wd={"type": "l2", "scale": 5e-4},
                           in_channel_num=32,
                           out_channel_num=64,
                           pool_size=[5, 5],
                           pool_stride=[2, 2],
                           activation={"type": "relu"},
                           activation_before_pooling=activation_before_pooling,
                           bn=bn))

    brain.attach(cnn_block(ksize=None,
                           initial_bias_value=0.1,
                           init_para={"name": "truncated_normal",
                                      "stddev": 0.1},
                           wd={"type": "l2", "scale": 5e-4},
                           in_channel_num=3136,
                           out_channel_num=512,
                           activation={"type": "relu"},
                           bn=bn,
                           keep_prob=0.5))

    brain.attach(cnn_block(ksize=None,
                           initial_bias_value=0.1,
                           init_para={"name": "truncated_normal",
                                      "stddev": 0.1},
                           wd={"type": "l2", "scale": 5e-4},
                           in_channel_num=512,
                           out_channel_num=10,
                           bn=bn,
                           activation=None))

    brain.attach(SoftmaxWithLossLayer(
        class_num=10,
        inputs=[{"name": "ip4", "idxs": [0]},
                {"name": "system_in", "idxs": [1]}],
        name="loss"))

    source = MNISTFeedSource(name="MNIST",
                             url='http://yann.lecun.com/exdb/mnist/',
                             work_dir=AKID_DATA_PATH + '/mnist',
                             num_train=50000,
                             num_val=5000,
                             center=True,
                             scale=True)

    sensor = FeedSensor(name='data',
                        source_in=source,
                        batch_size=64,
                        val_batch_size=100)
    kid = Kid(
        sensor,
        brain,
        MomentumKongFu(
            lr_scheme={
                "name": LearningRateScheme.exp_decay,
                "base_lr": 0.01,
                "decay_rate": 0.95,
                "num_batches_per_epoch": 468,
                "decay_epoch_num": 1},
            momentum=0.9),
        max_steps=4000)
    kid.setup()

    return kid


if __name__ == "__main__":
    kid = setup()
    kid.practice()
