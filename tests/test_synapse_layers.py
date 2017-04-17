import numpy as np

from akid.utils.test import AKidTestCase, main, TestFactory
from akid.sugar import cnn_block
from akid import sugar
from akid.layers import SoftmaxWithLossLayer
from akid.layers import ConvolutionLayer
from akid import backend as A


class TestSynapseLayers(AKidTestCase):
    def setUp(self):
        super(TestSynapseLayers, self).setUp()
        sugar.init()

    def test_conv_backward(self):
        filter = np.array([[
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]],
            [[2, 2, 2],
             [2, 2, 2],
             [2, 2, 2]]
        ]],
                          dtype=np.float32)
        X_in = np.array([[[[1, 0],
                         [0, 2]]]],
                        dtype=np.float32)
        X_in = np.einsum('nchw->nhwc', X_in)
        in_channel_1_ref = np.array([[1, 1, 1, 0],
                              [1, 1, 1, 0],
                              [1, 1, 1, 0],
                                     [0, 0, 0, 0],],
                                    dtype=np.float32) +\
            np.array([[0, 0, 0, 0],
                      [0, 2, 2, 2],
                      [0, 2, 2, 2],
                      [0, 2, 2, 2],],
                     dtype=np.float32)
        in_channel_2_ref = in_channel_1_ref * 2
        X_out_ref = np.array([in_channel_1_ref, in_channel_2_ref])
        X_out_ref = np.array([X_out_ref])
        X_out_ref = np.einsum('nchw->nhwc', X_out_ref)
        # Convert to H X W X IN_CHANNEL X OUT_CHANNEL
        filter = np.einsum('oihw->hwio', filter)

        l = ConvolutionLayer(ksize=[3, 3],
                             strides=[1, 1, 1, 1],
                             out_channel_num=1,
                             padding="VALID",
                             initial_bias_value=1.,
                             init_para={"name": "tensor",
                                        "value": filter},
                             name="test_backward_conv")
        # TODO: Ideally no forward should be needed, but now (2b1416) setup is
        # coupled with forward, so we do forward here. What is forwarded does
        # not matter, since we do not need to use them.
        l.forward(A.Tensor(X_out_ref))
        X_out = l.backward(X_in)

        A.init()
        X_out_eval = A.eval(X_out)
        assert (X_out_eval == X_out_ref).all()

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
        kid = TestFactory.get_test_kid(source, brain)
        kid.setup()

        loss = kid.practice()
        assert loss < 13


if __name__ == "__main__":
    main()
