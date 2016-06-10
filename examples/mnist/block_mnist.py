from akid.sugar.block_templates import cnn_block
from akid import AKID_DATA_PATH
from akid import Brain, MNISTFeedSource, FeedSensor, Survivor, MomentumKongFu


def setup(bn=None, activation_before_pooling=False):
    brain = Brain(name="sugar_mnist")

    brain.attach(cnn_block(ksize=[5, 5],
                           initial_bias_value=0.,
                           init_para={"name": "truncated_normal",
                                      "stddev": 0.1},
                           wd={"type": "l2", "scale": 5e-4},
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
                           out_channel_num=512,
                           activation={"type": "relu"},
                           bn=bn,
                           keep_prob=0.5))

    brain.attach(cnn_block(ksize=None,
                           initial_bias_value=0.1,
                           init_para={"name": "truncated_normal",
                                      "stddev": 0.1},
                           wd={"type": "l2", "scale": 5e-4},
                           out_channel_num=10,
                           bn=bn,
                           activation={"type": "softmax"}))

    source = MNISTFeedSource(name="MNIST",
                             url='http://yann.lecun.com/exdb/mnist/',
                             work_dir=AKID_DATA_PATH + '/mnist',
                             num_train=50000,
                             num_val=5000,
                             center=True,
                             scale=True)

    kid = Survivor(FeedSensor(name='data',
                              source_in=source,
                              batch_size=64,
                              val_batch_size=100),
                   brain,
                   MomentumKongFu(momentum=0.9,
                                  base_lr=0.01,
                                  decay_rate=0.95,
                                  decay_epoch_num=1),
                   max_steps=4000)
    kid.setup()

    return kid


if __name__ == "__main__":
    kid = setup()
    kid.practice()
