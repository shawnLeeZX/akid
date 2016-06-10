import time

from akid.tests.test import TestCase, main, TestFactory
from akid import Brain
from akid.sugar import cnn_block


class TestActivationLayers(TestCase):
    def test_group_softmax(self):
        """
        This is to test initialization could be properly set up. It does not
        involve run time things.
        """
        brain = Brain(name="test_brain")
        brain.attach(cnn_block(ksize=[5, 5],
                               initial_bias_value=0.,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l2", "scale": 5e-4},
                               out_channel_num=32,
                               pool_size=[5, 5],
                               pool_stride=[5, 5],
                               activation={"type": "gsoftmax",
                                           "group_size": 2}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.1,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l2", "scale": 5e-4},
                               out_channel_num=512,
                               activation={"type": "gsoftmax",
                                           "group_size": 8}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.1,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l2", "scale": 5e-4},
                               out_channel_num=10,
                               activation={"type": "softmax"}))

        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_survivor(source, brain)
        kid.setup()

        precision = kid.practice()
        assert precision > 0.9

    def test_linearization(self):
        """
        This is to test initialization could be properly set up. It does not
        involve run time things.
        """
        brain = Brain(name="test_brain")
        brain.attach(cnn_block(ksize=[5, 5],
                               initial_bias_value=0.,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l2", "scale": 5e-4},
                               out_channel_num=32,
                               pool_size=[5, 5],
                               pool_stride=[5, 5],
                               activation={"type": "linearize",
                                           "group_size": 2}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.1,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l2", "scale": 5e-4},
                               out_channel_num=512,
                               activation={"type": "linearize",
                                           "group_size": 8}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.1,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l2", "scale": 5e-4},
                               out_channel_num=10,
                               activation={"type": "softmax"}))

        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_survivor(source, brain)
        kid.setup()

        start_time = time.time()
        precision = kid.practice()
        end_time = time.time()
        assert precision > 0.9
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
                               activation={"type": "softmax"},
                               bn={"gamma_init": 1., "share_gamma": True}))

        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_survivor(source, brain)
        kid.setup()

        precision = kid.practice()
        assert precision > 0.7

if __name__ == "__main__":
    main()
