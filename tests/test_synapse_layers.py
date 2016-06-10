from akid.tests.test import TestCase, main, TestFactory
from akid import Brain
from akid.sugar import cnn_block


class TestSynapseLayers(TestCase):
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
                               activation={"type": "linearize",
                                           "group_size": 2}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.1,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l1", "scale": 0.0005},
                               out_channel_num=512,
                               activation={"type": "linearize",
                                           "group_size": 8}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.1,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l1", "scale": 0.0005},
                               out_channel_num=10,
                               activation={"type": "softmax"}))

        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_survivor(source, brain)
        kid.setup()

        precision = kid.practice()
        assert precision > 0.7

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
                               activation={"type": "softmax"}))

        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_survivor(source, brain)
        kid.setup()

        precision = kid.practice()
        assert precision > 0.98

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
                               activation={"type": "softmax"}))

        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_survivor(source, brain)
        kid.setup()

        precision = kid.practice()
        assert precision > 0.97

if __name__ == "__main__":
    main()
