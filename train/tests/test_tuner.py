from akid.train.tuner import tune
from akid.tests.test import TestCase, main


net_paras_list = []
net_paras_list.append({
    "activation": [
        {"type": "relu"},
        {"type": "relu"},
        {"type": "relu"},
        {"type": "relu"}],
    "bn": {"gamma_init": 1, "fix_gamma": True}})
net_paras_list.append({
    "activation": [
        {"type": "relu"},
        {"type": "relu"},
        {"type": "relu"},
        {"type": "relu"}],
    "bn": None})

opt_paras_list = []
opt_paras_list.append({"lr": 0.025})
opt_paras_list.append({"lr": 0.05})


def setup(graph):
    from akid import AKID_DATA_PATH
    from akid import Brain, MNISTFeedSource, FeedSensor, Kid
    from akid import MomentumKongFu
    from akid.layers import DropoutLayer
    from akid.sugar import cnn_block

    brain = Brain(name="one-layer-mnist")

    brain.attach(DropoutLayer(keep_prob=0.8, name='dropout0'))

    brain.attach(cnn_block(
        ksize=[5, 5],
        init_para={
            "name": "truncated_normal",
            "stddev": 0.1},
        wd={"type": "l2", "scale": 0.0005},
        out_channel_num=32,
        pool_size=[5, 5],
        pool_stride=[5, 5],
        activation={{ net_paras["activation"][0] }},
        keep_prob=0.5,
        bn={{ net_paras["bn"] }}))

    brain.attach(cnn_block(
        init_para={
            "name": "truncated_normal",
            "stddev": 0.1},
        wd={"type": "l2", "scale": 0.0005},
        out_channel_num=10,
        activation={"type": "softmax"},
        bn={{ net_paras["bn"] }}))

    # Set up a sensor.
    # #########################################################################
    source = MNISTFeedSource(name="MNIST",
                             url='http://yann.lecun.com/exdb/mnist/',
                             work_dir=AKID_DATA_PATH + '/mnist',
                             num_train=50000,
                             num_val=5000,
                             center=True,
                             scale=True)

    kid = Kid(FeedSensor(name='data',
                              source_in=source,
                              batch_size=64,
                              val_batch_size=100),
                   brain,
                   MomentumKongFu(momentum=0.9,
                                  base_lr={{ opt_paras["lr"] }},
                                  decay_rate=0.95,
                                  decay_epoch_num=1),
                   max_steps=1000,
                   graph=graph)
    kid.setup()
    return kid


class TestTuner(TestCase):
    def test_tuner(self):
        precision_list = tune(setup, opt_paras_list, net_paras_list)
        print precision_list
        # TODO(Shuai): When one training instance fails and quit, its precision
        # will not be added, so the test here cannot check out that
        # case. Specifically, When memory exhausts, this test will still pass.
        for precision in precision_list:
            assert precision > 0.80


if __name__ == "__main__":
    main()
