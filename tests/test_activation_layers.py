import tensorflow as tf
import numpy as np
from akid.utils import glog as log

from akid.utils.test import AKidTestCase, main, TestFactory
from akid import Brain
from akid.sugar import cnn_block
from akid import sugar
from akid.layers import SoftmaxWithLossLayer


log.init()


class TestActivationLayers(AKidTestCase):
    def setUp(self):
        super(TestActivationLayers, self).setUp()
        sugar.init()

    def test_softmax_normalization(self):
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

    def test_gsmax_tensor_input(self):
        from akid.layers import GroupSoftmaxLayer
        from math import log

        input = tf.constant([log(3), log(3), log(3),
                             log(3), log(3), log(3)])
        layer = GroupSoftmaxLayer(group_size=3, name="gsmax")
        layer.setup(input)

        with tf.Session():
            out = layer.data.eval()
            out_ref = np.array([0.2560102,  0.2560102,  0.2560102,
                                0.2560102,  0.2560102,  0.2560102])
            assert np.sum(abs(out - out_ref)) <= 10e-4

    def test_gsmax_list_input(self):
        from akid.layers import GroupSoftmaxLayer
        from math import log

        input = [tf.constant([log(3), log(3), log(3)]),
                 tf.constant([log(3), log(3), log(3)])]
        layer = GroupSoftmaxLayer(group_size=3, name="gsmax")
        layer.setup(input)

        with tf.Session():
            out = layer.data.eval()
            out_ref = np.array([0.2560102,  0.2560102,  0.2560102,
                                0.2560102,  0.2560102,  0.2560102])
            assert np.sum(abs(out - out_ref)) <= 10e-4

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

        # Test input as tensor
        # #################################################################

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

        # Test input as list
        # ###############################################################
        # Test Maxout
        layer = CollapseOutLayer(group_size=2, type="maxout", name="maxout")
        input = [tf.constant([1., 0.]), tf.constant([2., 0.])]
        layer.setup(input)
        with tf.Session():
            output = layer.data.eval()
            out_ref = np.array([1., 2.])
            assert np.sum(abs(output - out_ref)) <= 1e-4,\
                "output: {}, out_ref {}.".format(output, out_ref)

        # Test Average out
        layer = CollapseOutLayer(group_size=2,
                               type="average_out",
                               name="average_out")
        layer.setup(input)
        with tf.Session():
            output = layer.data.eval()
            out_ref = np.array([0.5, 1.])
            assert np.sum(abs(output - out_ref)) <= 1e-4,\
                "output: {}, out_ref {}.".format(output, out_ref)


if __name__ == "__main__":
    main()
