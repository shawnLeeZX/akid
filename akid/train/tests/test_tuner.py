from akid.train.tuner import tune
from akid.utils.test import TestCase, main


net_paras_list = []
net_paras_list.append({
    "activation": [
        {"type": "relu"},
        {"type": "relu"},
        {"type": "relu"},
        {"type": "relu"}],
    })
net_paras_list.append({
    "activation": [
        {"type": "relu"},
        {"type": "relu"},
        {"type": "relu"},
        {"type": "relu"}],
    })

opt_paras_list = []
opt_paras_list.append({"lr": 0.025, "engine": {"name": "single"}})
opt_paras_list.append({"lr": 0.05, "engine": {"name": "data_parallel",
                                              "num_gpu": 2}})


def setup():
    from akid import AKID_DATA_PATH
    from akid import GraphBrain, MNISTFeedSource, FeedSensor, Kid
    from akid import MomentumKongFu
    from akid.layers import DropoutLayer, SoftmaxWithLossLayer
    from akid.sugar import cnn_block
    from akid import LearningRateScheme

    brain = GraphBrain(name="one-layer-mnist")

    brain.attach(DropoutLayer(keep_prob=0.8, name='dropout0'))

    brain.attach(cnn_block(
        ksize=[5, 5],
        init_para={
            "name": "truncated_normal",
            "stddev": 0.1},
        wd={"type": "l2", "scale": 0.0005},
        in_channel_num=1,
        out_channel_num=32,
        pool_size=[5, 5],
        pool_stride=[5, 5],
        activation={{ net_paras["activation"][0] }},
        keep_prob=0.5,
        ))

    brain.attach(cnn_block(
        init_para={
            "name": "truncated_normal",
            "stddev": 0.1},
        wd={"type": "l2", "scale": 0.0005},
        in_channel_num=1152,
        out_channel_num=10,
        activation=None,
        ))

    brain.attach(SoftmaxWithLossLayer(
        class_num=10,
        inputs=[{"name": brain.get_last_layer_name()},
                {"name": "system_in",
                 "idxs": [1]}],
        name="softmax"))

    # Set up a sensor.
    # #########################################################################
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

    kid = Kid(sensor,
              brain,
              MomentumKongFu(momentum=0.9,
                             lr_scheme={
                                 "name": LearningRateScheme.exp_decay,
                                 "base_lr": {{ opt_paras["lr"] }},
                                 "decay_rate": 0.95,
                                 "num_batches_per_epoch": sensor.num_batches_per_epoch,
                                 "decay_epoch_num": 1}),
              engine={{ opt_paras["engine"] }},
              max_steps=1000)
    kid.setup()
    return kid


class TestTuner(TestCase):
    def test_tuner(self):
        # NOTE: No assertion check. Have to read the log.
        tune(setup,
             opt_paras_list,
             net_paras_list,
             gpu_num_per_instance=[1, 1, 2, 2])


if __name__ == "__main__":
    main()
