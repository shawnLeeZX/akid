from __future__ import absolute_import
from akid.train.tuner import tune


net_paras_list = []
net_paras_list.append({
    "activation": [
        {"type": "relu"},
        {"type": "relu"},
        {"type": "relu"},
        {"type": "relu"}],
    "bn": True})
net_paras_list.append({
    "activation": [
        {"type": "maxout", "group_size": 2},
        {"type": "maxout", "group_size": 2},
        {"type": "maxout", "group_size": 2},
        {"type": "maxout", "group_size": 5}],
    "bn": True})
net_paras_list.append({
    "activation": [
        {"type": "gsoftmax", "group_size": 2},
        {"type": "gsoftmax", "group_size": 2},
        {"type": "gsoftmax", "group_size": 2},
        {"type": "gsoftmax", "group_size": 5}],
    "bn": True})

opt_paras_list = []
opt_paras_list.append({"lr": 0.025})
opt_paras_list.append({"lr": 0.05})


def setup(graph):
    from akid import AKID_DATA_PATH
    from akid import GraphBrain, Cifar10FeedSource, FeedSensor, Kid
    from akid import MomentumKongFu
    from akid.layers import DropoutLayer
    from akid.sugar import cnn_block

    brain = GraphBrain(name="maxout")

    brain.attach(DropoutLayer(keep_prob=0.8, name='dropout0'))

    brain.attach(cnn_block(
        ksize=[8, 8],
        init_para={
            "name": "uniform",
            "range": 0.005},
        wd={"type": "l2", "scale": 0.0005},
        out_channel_num=192,
        pool_size=[4, 4],
        pool_stride=[2, 2],
        activation={{ net_paras["activation"][0] }},
        keep_prob=0.5,
        bn={{ net_paras["bn"] }}))

    brain.attach(cnn_block(
        ksize=[8, 8],
        init_para={
            "name": "uniform",
            "range": 0.005},
        wd={"type": "l2", "scale": 0.0005},
        out_channel_num=384,
        pool_size=[4, 4],
        pool_stride=[2, 2],
        activation={{ net_paras["activation"][1] }},
        keep_prob=0.5,
        bn={{ net_paras["bn"] }}))

    brain.attach(cnn_block(
        ksize=[5, 5],
        init_para={
            "name": "uniform",
            "range": 0.005},
        wd={"type": "l2", "scale": 0.0005},
        out_channel_num=384,
        pool_size=[2, 2],
        pool_stride=[2, 2],
        activation={{ net_paras["activation"][2] }},
        keep_prob=0.5,
        bn={{ net_paras["bn"] }}))

    brain.attach(cnn_block(
        init_para={
            "name": "uniform",
            "range": 0.005},
        wd={"type": "l2", "scale": 0.004},
        out_channel_num=2500,
        activation={{ net_paras["activation"][3] }},
        keep_prob=0.5,
        bn={{ net_paras["bn"] }}))

    brain.attach(cnn_block(
        init_para={
            "name": "uniform",
            "range": 0.005},
        wd={"type": "l2", "scale": 0.},
        out_channel_num=10,
        activation={"type": "softmax"},
        bn={{ net_paras["bn"] }}))

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
        MomentumKongFu(base_lr={{ opt_paras["lr"] }},
                       momentum=0.5,
                       decay_rate=0.1,
                       decay_epoch_num=50),
        max_steps=200000,
        graph=graph,
    )

    survivor.setup()
    return survivor


if __name__ == "__main__":
    tune(setup, opt_paras_list, net_paras_list)
