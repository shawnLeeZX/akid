from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from akid.utils import glog as log

from akid.utils.test import AKidTestCase, main, TestFactory, skipUnless
from akid import GraphBrain
from akid.sugar import cnn_block
from akid import sugar
from akid.layers import SoftmaxWithLossLayer, BatchNormalizationLayer
from akid import backend as A
from six.moves import zip


log.init()


class TestActivationLayers(AKidTestCase):
    def setUp(self):
        super(TestActivationLayers, self).setUp()
        sugar.init()
        A.reset()

    @skipUnless(A.backend() == A.TF)
    def test_softmax_normalization(self):
        brain = GraphBrain(name="test_brain")
        brain.attach(cnn_block(ksize=[5, 5],
                               initial_bias_value=0.,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l2", "scale": 5e-4},
                               in_channel_num=1,
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
                               in_channel_num=1152,
                               out_channel_num=512,
                               activation={"type": "ngsmax",
                                           "group_size": 8}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.1,
                               init_para={"name": "truncated_normal",
                                          "stddev": 0.1},
                               wd={"type": "l2", "scale": 5e-4},
                               in_channel_num=512,
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

    @skipUnless(A.backend() == A.TF)
    def test_gsmax_tensor_input(self):
        from akid.layers import GroupSoftmaxLayer
        from math import log

        input = tf.constant([log(3), log(3), log(3),
                             log(3), log(3), log(3)])
        layer = GroupSoftmaxLayer(group_size=3, name="gsmax")
        layer.forward(input)

        with tf.Session():
            out = layer.data.eval()
            out_ref = np.array([0.2560102,  0.2560102,  0.2560102,
                                0.2560102,  0.2560102,  0.2560102])
            assert np.sum(abs(out - out_ref)) <= 10e-4

    @skipUnless(A.backend() == A.TF)
    def test_gsmax_list_input(self):
        from akid.layers import GroupSoftmaxLayer
        from math import log

        input = [tf.constant([log(3), log(3), log(3)]),
                 tf.constant([log(3), log(3), log(3)])]
        layer = GroupSoftmaxLayer(group_size=3, name="gsmax")
        layer.forward(input)

        with tf.Session():
            out = layer.data.eval()
            out_ref = np.array([0.2560102,  0.2560102,  0.2560102,
                                0.2560102,  0.2560102,  0.2560102])
            assert np.sum(abs(out - out_ref)) <= 10e-4

    def test_bn_2d(self):
        l = BatchNormalizationLayer(10, 0, 1, name="test_bn")
        l.setup()
        X_in = np.random.randn(10, 10)
        self.assertNdarrayNotAlmostEquals(X_in.mean(0), 0)
        self.assertNdarrayNotAlmostEquals(X_in.std(0), 1)
        X_in = A.Tensor(X_in)
        X_out = l(X_in)
        A.init()
        X_out_eval = A.eval(X_out)
        gamma = A.eval(l.weights)
        self.assertNdarrayAlmostEquals(X_out_eval.mean(0), 0)
        # The normalization only holds roughly, so do not do a strong assertion
        # on the equality.
        self.assertNdarrayAlmostEquals(X_out_eval.std(0), gamma, places=4)

    def test_bn_4d(self):
        l = BatchNormalizationLayer(10, 0, 1, name="test_bn")
        l.setup()
        X_in = np.random.randn(10, 10, 10, 10)
        if A.DATA_FORMAT == "CHW":
            reduce_indices = (0, 2, 3)
        elif A.DATA_FORMAT == "HWC":
            reduce_indices = (0, 1, 2)
        else:
            raise ValueError("Data Format {} is not supported".format(A.DATA_FORMAT))
        self.assertNdarrayNotAlmostEquals(X_in.mean(reduce_indices), 0)
        self.assertNdarrayNotAlmostEquals(X_in.std(reduce_indices), 1)
        X_in = A.Tensor(X_in)
        X_out = l(X_in)
        A.init()
        X_out_eval = A.eval(X_out)
        gamma = A.eval(l.weights)
        self.assertNdarrayAlmostEquals(X_out_eval.mean(reduce_indices), 0)
        # The normalization only holds roughly, so do not do a strong assertion
        # on the equality.
        self.assertNdarrayAlmostEquals(X_out_eval.std(reduce_indices), gamma, places=4)

    @skipUnless(A.backend() == A.TF)
    def test_reduce_out(self):
        from akid.layers import CollapseOutLayer

        input = tf.constant([1., 0.])

        # Test input as tensor
        # #################################################################

        # Test Maxout
        layer = CollapseOutLayer(group_size=2, type="maxout", name="maxout")
        layer.forward(input)
        with tf.Session():
            output = layer.data.eval()
            assert output == 1

        # Test Average out
        layer = CollapseOutLayer(group_size=2,
                               type="average_out",
                               name="average_out")
        layer.forward(input)
        with tf.Session():
            output = layer.data.eval()
            assert output == 0.5

        # Test input as list
        # ###############################################################
        # Test Maxout
        layer = CollapseOutLayer(group_size=2, type="maxout", name="maxout")
        input = [tf.constant([1., 0.]), tf.constant([2., 0.])]
        layer.forward(input)
        with tf.Session():
            output = layer.data.eval()
            out_ref = np.array([1., 2.])
            assert np.sum(abs(output - out_ref)) <= 1e-4,\
                "output: {}, out_ref {}.".format(output, out_ref)

        # Test Average out
        layer = CollapseOutLayer(group_size=2,
                               type="average_out",
                               name="average_out")
        layer.forward(input)
        with tf.Session():
            output = layer.data.eval()
            out_ref = np.array([0.5, 1.])
            assert np.sum(abs(output - out_ref)) <= 1e-4,\
                "output: {}, out_ref {}.".format(output, out_ref)

    @skipUnless(A.backend() == A.TF)
    def test_relu_backward(self):
        X_forward_in = A.Tensor(np.array([1, -1], dtype=np.float32))
        X_backward_in = A.Tensor(np.array([2, 3], dtype=np.float32))

        X_backward_out_ref = np.array([2, 0], dtype=np.float32)

        from akid.layers import ReLULayer
        l = ReLULayer(name="relu")
        l.forward(X_forward_in)
        X_backward_out = l.backward(X_backward_in)

        A.init()
        X_backward_out_eval = A.eval(X_backward_out)
        assert (X_backward_out_eval == X_backward_out_ref).all(),\
            "output: {}, out_ref: {}.".format(X_backward_out_eval, X_backward_out_ref)

    @skipUnless(A.backend() == A.TF)
    def test_max_pooling(self):
        X_in = np.array([[[1., 2],
                          [3, 4]],
                         [[5, 6],
                          [7, 8]]], dtype=np.float32)
        X_in = np.einsum("chw->hwc", X_in)
        X_in = np.expand_dims(X_in, axis=0)

        X_out_ref = np.array([[[4.]], [[8]]], dtype=np.float32)
        X_out_ref = np.einsum("chw->hwc", X_out_ref)
        X_out_ref = np.expand_dims(X_out_ref, axis=1)

        A.init()

        # Test forward
        # #################################################################
        from akid.layers import MaxPoolingLayer
        l = MaxPoolingLayer(ksize=[2, 2],
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="max_pool")
        X_out = l.forward(X_in)
        X_out_eval = A.eval(X_out)
        assert (X_out_eval == X_out_ref).all(),\
            "X_out_eval: {}; X_out_ref: {}".format(X_out_eval, X_out_ref)

        from akid.layers import MaxPoolingLayer
        l = MaxPoolingLayer(ksize=[2, 2],
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            get_argmax_idx=True,
                            name="max_pool")
        X_out = l.forward(X_in)

        X_out_eval, X_out_indices_eval = A.eval([X_out, l.in_group_indices])

        X_out_indices_ref = np.array([[[6]], [[7]]], dtype=np.int)
        X_out_indices_ref = np.einsum("chw->hwc", X_out_indices_ref)
        X_out_indices_ref = np.expand_dims(X_out_indices_ref, axis=1)

        assert (X_out_eval == X_out_ref).all(),\
            "X_out_eval: {}; X_out_ref: {}".format(X_out_eval, X_out_ref)
        assert (X_out_indices_eval == X_out_indices_ref).all(),\
            "X_out_indices_eval: {}; X_out_indices_ref: {}".format(X_out_indices_eval, X_out_indices_ref)

        # Test backward
        # #################################################################
        X_out_b_ref = np.array([[[0, 0],
                               [0, 4]],
                              [[0, 0],
                               [0, 8]]], dtype=np.float32)
        X_out_b_ref = np.einsum("chw->hwc", X_out_b_ref)
        X_out_b_ref = np.expand_dims(X_out_b_ref, axis=0)

        X_in_b = np.array([[[4.]], [[8]]], dtype=np.float32)
        X_in_b = np.einsum("chw->hwc", X_in_b)
        X_in_b = np.expand_dims(X_in_b, axis=0)

        X_out_b = l.backward(A.Tensor(X_in_b))

        X_out_g_eval = A.eval(X_out_b)
        assert (X_out_g_eval == X_out_b_ref).all(),\
            "X_out_eval: {}; X_out_ref: {}".format(X_out_g_eval, X_out_b_ref)

        # Test list pooling
        # ###############################################################
        X_in = [A.Tensor(X_in)] * 2
        X_out_ref = [X_out_ref] * 2
        X_out_indices_ref = [X_out_indices_ref] * 2
        X_out = l.forward(X_in)
        X_out_eval = A.eval(X_out)
        l_eval = A.eval(l.in_group_indices)
        for X, X_ref in zip(X_out_eval, X_out_ref):
            assert (X == X_ref).all(),\
                "X_out_eval: {}; X_out_ref: {}".format(X, X_ref)
        for I, I_ref in zip(l_eval, X_out_indices_ref):
            assert (I == I_ref).all(),\
                "I_out_eval: {}; I_out_ref: {}".format(I, I_ref)

    @skipUnless(A.backend() == A.TF)
    def test_colorization_relu(self):
        F = np.array([
            [
                [
                    [1, -1],
                    [1, -1],
                ],
                [
                    [1, -1],
                    [1, -1]
                ]],
            [
                [
                    [-1, 1],
                    [-1, 1],
                ],
                [
                    [-1, 1],
                    [-1, 1]
                ]]
        ])
        F = np.einsum("nchw->nhwc", F)
        F = A.Tensor(F)
        C = np.array([
            [
                [[1, 1],
                 [1, 1]],
                [[0, 0],
                 [0, 0]],
                [[0, 0],
                 [0, 0]]
            ],
            [
                [[1, 1],
                 [1, 1]],
                [[0, 0],
                 [0, 0]],
                [[0, 0],
                 [0, 0]]
            ]
        ])
        C = np.einsum("nchw->nhwc", C)
        C = A.Tensor(C)
        X_out_ref = np.array([
            [
                [
                    [1, 0],
                    [1, 0],
                ],
                [
                    [1, 0],
                    [1, 0]
                ]],
            [
                [
                    [0, 1],
                    [0, 1],
                ],
                [
                    [0, 1],
                    [0, 1]
                ]]
        ])
        X_out_zero_channel = X_out_ref * 0
        X_out_ref = np.concatenate([X_out_ref] + [X_out_zero_channel] * 2, axis=1)
        X_out_ref = np.einsum("nchw->nhwc", X_out_ref)
        from akid.layers import ColorizationReLULayer
        l = ColorizationReLULayer(name="crelu")
        X_out = l.forward([F, C])
        A.init()
        X_out_eval = A.eval(X_out)
        assert (X_out_eval == X_out_ref).all(), \
            "X_out_eval: {}; X_out_ref: {}".format(X_out_eval, X_out_ref)

    @skipUnless(A.backend() == A.TORCH, "tf backend does not have nn.l2_norm function yet.")
    def test_l2norm_layer(self):
        from akid.layers import L2Norm

        X_in = np.array(
            [
                [1, 1],
                [2, 2]
            ]
        )
        l = L2Norm()
        X_out = l(A.Tensor(X_in))
        X_out = A.eval(X_out)
        for i in range(X_out.shape[0]):
            self.assertAlmostEquals(np.linalg.norm(X_out[i, :]), 1, places=6)


if __name__ == "__main__":
    main()
