from akid.tests.test import TestCase, main, TestFactory
from akid import Brain
from akid.sugar import cnn_block
from akid import sugar
from akid.layers import SoftmaxWithLossLayer


class TestSynapseLayers(TestCase):
    def setUp(self):
        sugar.init()

    def test_l1_regularization(self):
        brain = Brain(name="test_brain")
        brain.attach(cnn_block(ksize=[5, 5],
                               initial_bias_value=0.,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l1", "scale": 0.0005},
                               out_channel_num=32,
                               pool_size=[5, 5],
                               pool_stride=[5, 5],
                               activation={"type": "gsmax",
                                           "group_size": 2}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.1,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l1", "scale": 0.0005},
                               out_channel_num=512,
                               activation={"type": "gsmax",
                                           "group_size": 8}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.1,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l1", "scale": 0.0005},
                               out_channel_num=10,
                               activation=None))
        brain.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[{"name": "ip3", "idxs": [0]},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))

        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_survivor(source, brain)
        kid.setup()

        loss = kid.practice()
        assert loss < 13

    def test_uniform_unit_scale_initializer(self):
        brain = Brain(name="test_brain")
        brain.attach(cnn_block(ksize=[5, 5],
                               initial_bias_value=0.,
                               init_para={"name": "uniform_unit_scaling",
                                          "factor": 1.43},
                               wd={"type": "l2", "scale": 0.0005},
                               out_channel_num=32,
                               pool_size=[5, 5],
                               pool_stride=[5, 5],
                               activation={"type": "relu"}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.,
                               init_para={"name": "uniform_unit_scaling",
                                          "factor": 1.43},
                               wd={"type": "l2", "scale": 0.0005},
                               out_channel_num=512,
                               activation={"type": "relu"}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.,
                               init_para={"name": "uniform_unit_scaling",
                                          "factor": 1.43},
                               wd={"type": "l2", "scale": 0.0005},
                               out_channel_num=10,
                               activation=None))
        brain.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[{"name": "ip3", "idxs": [0]},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))

        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_survivor(source, brain)
        kid.setup()

        loss = kid.practice()
        assert loss < 0.5

    def test_unit_gradient_initializer(self):
        brain = Brain(name="test_brain")
        brain.attach(cnn_block(ksize=[5, 5],
                               initial_bias_value=0.,
                               init_para={"name": "msra_init"},
                               wd={"type": "l2", "scale": 0.0005},
                               out_channel_num=32,
                               pool_size=[5, 5],
                               pool_stride=[5, 5],
                               activation={"type": "relu"}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.,
                               init_para={"name": "msra_init"},
                               wd={"type": "l2", "scale": 0.0005},
                               out_channel_num=512,
                               activation={"type": "relu"}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.,
                               init_para={"name": "msra_init"},
                               wd={"type": "l2", "scale": 0.0005},
                               out_channel_num=10,
                               activation=None))
        brain.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[{"name": "ip3", "idxs": [0]},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))

        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_survivor(source, brain)
        kid.setup()

        loss = kid.practice()
        assert loss < 1

if __name__ == "__main__":
    main()
