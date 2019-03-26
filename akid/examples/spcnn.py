from __future__ import absolute_import
from six.moves import range
def setup(graph=None):

    from akid import GraphBrain
    from akid.sugar import cnn_block
    from akid import AKID_DATA_PATH
    from akid import Cifar10FeedSource, FeedSensor, Kid
    from akid import MomentumKongFu

    brain = GraphBrain(name="spcnn")

    out_channel_num_list = [64, 128, 256, 512]
    group_size_list = [2, 4, 8, 16]

    for i in range(0, 4):
        brain.attach(cnn_block(
            ksize=[3, 3],
            init_para={
                "name": "uniform",
                "range": 0.005},
            wd={"type": "l2", "scale": 0.005},
            out_channel_num=out_channel_num_list[i],
            pool_size=[2, 2],
            pool_stride=[2, 2],
            activation={"type": "linearize", "group_size": group_size_list[i]},
            keep_prob=0.5,
            bn={"gamma_init": 1, "fix_gamma": True}))

    brain.attach(cnn_block(
        init_para={
            "name": "uniform",
            "range": 0.005},
        wd={"type": "l2", "scale": 0.005},
        out_channel_num=10,
        activation={"type": "softmax"},
        bn={"gamma_init": 1, "fix_gamma": True}))

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
    survivor = Kid(
        sensor,
        brain,
        MomentumKongFu(base_lr=1.0,
                       momentum=0.5,
                       decay_rate=0.1,
                       decay_epoch_num=25),
        max_steps=60000,
        graph=graph,
    )

    survivor.setup()
    return survivor

kid = setup()
kid.practice()
