import time

import tensorflow as tf

from akid.tests.test import TestCase, main, TestFactory
from akid import Brain
from akid.sugar import cnn_block
from akid import sugar
from akid.layers import SoftmaxWithLossLayer


class TestActivationLayers(TestCase):
    def setUp(self):
        sugar.init()

    def test_group_softmax(self):
        brain = Brain(name="test_brain")
        brain.attach(cnn_block(ksize=[5, 5],
                               initial_bias_value=0.,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l2", "scale": 5e-4},
                               out_channel_num=32,
                               pool_size=[5, 5],
                               pool_stride=[5, 5],
                               activation={"type": "ngsmax",
                                           "group_size": 2}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.1,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l2", "scale": 5e-4},
                               out_channel_num=512,
                               activation={"type": "ngsmax",
                                           "group_size": 8}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.1,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l2", "scale": 5e-4},
                               out_channel_num=10,
                               activation=None))
        brain.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[{"name": "ip3"},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))

        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_kid(source, brain)
        kid.setup()

        loss = kid.practice()
        assert loss < 1.5

    def test_linearization(self):
        brain = Brain(name="test_brain")
        brain.attach(cnn_block(ksize=[5, 5],
                               initial_bias_value=0.,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l2", "scale": 5e-4},
                               out_channel_num=32,
                               pool_size=[5, 5],
                               pool_stride=[5, 5],
                               activation={"type": "gsmax",
                                           "group_size": 2}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.1,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l2", "scale": 5e-4},
                               out_channel_num=512,
                               activation={"type": "gsmax",
                                           "group_size": 8}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.1,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l2", "scale": 5e-4},
                               out_channel_num=10,
                               activation=None))
        brain.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[{"name": "ip3"},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))

        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_kid(source, brain)
        kid.setup()

        start_time = time.time()
        loss = kid.practice()
        end_time = time.time()
        assert loss < 2
        assert end_time - start_time < 60

    def test_bn(self):
        brain = Brain(name="test_brain")
        brain.attach(cnn_block(ksize=[5, 5],
                               initial_bias_value=0.,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l2", "scale": 5e-4},
                               out_channel_num=32,
                               pool_size=[5, 5],
                               pool_stride=[5, 5],
                               activation={"type": "relu"},
                               bn={"gamma_init": 1., "share_gamma": True}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.1,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l2", "scale": 5e-4},
                               out_channel_num=512,
                               activation={"type": "relu"},
                               bn={"gamma_init": 1., "share_gamma": True}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.1,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l2", "scale": 5e-4},
                               out_channel_num=10,
                               activation=None,
                               bn={"gamma_init": 1., "share_gamma": True}))

        brain.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[{"name": "ip3", "idxs": [0]},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))

        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_kid(source, brain)
        kid.setup()

        loss = kid.practice()
        assert loss < 4

    def test_reduce_out(self):
        from akid.layers import CollapseOutLayer

        input = tf.constant([1., 0.])

        # Test Maxout
        layer = CollapseOutLayer(group_size=2, type="maxout", name="maxout")
        layer.setup(input)
        with tf.Session():
            output = layer.data.eval()
            assert output == 1

        # Test Average out
        layer = CollapseOutLayer(group_size=2,
                               type="average_out",
                               name="average_out")
        layer.setup(input)
        with tf.Session():
            output = layer.data.eval()
            assert output == 0.5

if __name__ == "__main__":
    main()
