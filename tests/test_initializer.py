from __future__ import division

from __future__ import absolute_import
import numpy as np

from akid.utils.test import AKidTestCase, main, TestFactory, skipUnless
from akid import initializers
from akid import GraphBrain
from akid.sugar import cnn_block
import akid.backend as A
from akid.layers import SoftmaxWithLossLayer, ConvolutionLayer


def build_kid(init_para):
    """
    Build a kid whose brain is initialized with different initialization
    methods.
    """
    brain = GraphBrain(name="test_brain")
    brain.attach(cnn_block(ksize=[5, 5],
                            initial_bias_value=0.,
                            init_para=init_para,
                            wd={"type": "l2", "scale": 0.0005},
                            in_channel_num=1,
                            out_channel_num=32,
                            pool_size=[5, 5],
                            pool_stride=[5, 5],
                            activation={"type": "relu"}))

    brain.attach(cnn_block(ksize=None,
                            initial_bias_value=0.,
                            init_para=init_para,
                            wd={"type": "l2", "scale": 0.0005},
                            in_channel_num=1152,
                            out_channel_num=512,
                            activation={"type": "relu"}))

    brain.attach(cnn_block(ksize=None,
                            initial_bias_value=0.,
                            init_para=init_para,
                            wd={"type": "l2", "scale": 0.0005},
                            in_channel_num=512,
                            out_channel_num=10,
                            activation=None))
    loss_layer_name = brain.get_last_layer_name()
    brain.attach(SoftmaxWithLossLayer(
        class_num=10,
        inputs=[{"name": loss_layer_name, "idxs": [0]},
                {"name": "system_in", "idxs": [1]}],
        name="loss"))

    sensor = TestFactory.get_test_sensor()
    kid = TestFactory.get_test_kid(sensor, brain)

    return kid


class TestInitializer(AKidTestCase):
    def test_tensor(self):
        X_np = np.array([[1., 1], [2, 3]])
        init = initializers.get("tensor", value=X_np)
        X = A.get_variable("X", initializer=init)
        A.init()
        X_eval = A.eval(X)
        assert (X_eval == X_np).all()

    def test_xavier(self):
        l = ConvolutionLayer(ksize=[3, 3],
                             in_channel_num=10,
                             out_channel_num=10,
                             init_para={"name": "xavier"},
                             name='test_xavier')
        l.setup()

        A.init()
        W = A.eval(l.weights)
        self.assertLess(abs(W.mean()), 0.04)
        self.assertLess(abs(W.std() - np.sqrt(2/(3*3*10 + 9))), 0.001)

    def test_xavier_full(self):
        kid = build_kid({"name": "xavier"})
        kid.setup()
        loss = kid.practice()
        self.assertLessEqual(loss, 0.3)

        kid.teardown()

    @skipUnless(A.backend() == A.TF)
    def test_uniform_unit_scale_initializer(self):
        brain = GraphBrain(name="test_brain")
        brain.attach(cnn_block(ksize=[5, 5],
                               initial_bias_value=0.,
                               init_para={"name": "uniform_unit_scaling",
                                          "factor": 1.43},
                               wd={"type": "l2", "scale": 0.0005},
                               in_channel_num=1,
                               out_channel_num=32,
                               pool_size=[5, 5],
                               pool_stride=[5, 5],
                               activation={"type": "relu"}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.,
                               init_para={"name": "uniform_unit_scaling",
                                          "factor": 1.43},
                               wd={"type": "l2", "scale": 0.0005},
                               in_channel_num=1152,
                               out_channel_num=512,
                               activation={"type": "relu"}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.,
                               init_para={"name": "uniform_unit_scaling",
                                          "factor": 1.43},
                               wd={"type": "l2", "scale": 0.0005},
                               in_channel_num=512,
                               out_channel_num=10,
                               activation=None))
        loss_layer_name = brain.get_last_layer_name()
        brain.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[{"name": loss_layer_name, "idxs": [0]},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))

        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_kid(source, brain)
        kid.setup()

        loss = kid.practice()
        assert loss < 0.5

    @skipUnless(A.backend() == A.TF)
    def test_unit_gradient_initializer(self):
        brain = GraphBrain(name="test_brain")
        brain.attach(cnn_block(ksize=[5, 5],
                               initial_bias_value=0.,
                               init_para={"name": "msra"},
                               wd={"type": "l2", "scale": 0.0005},
                               in_channel_num=1,
                               out_channel_num=32,
                               pool_size=[5, 5],
                               pool_stride=[5, 5],
                               activation={"type": "relu"}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.,
                               init_para={"name": "msra"},
                               wd={"type": "l2", "scale": 0.0005},
                               in_channel_num=1152,
                               out_channel_num=512,
                               activation={"type": "relu"}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.,
                               init_para={"name": "msra"},
                               wd={"type": "l2", "scale": 0.0005},
                               in_channel_num=512,
                               out_channel_num=10,
                               activation=None))
        loss_layer_name = brain.get_last_layer_name()
        brain.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[{"name": loss_layer_name, "idxs": [0]},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))

        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_kid(source, brain)
        kid.setup()

        loss = kid.practice()
        assert loss < 1

if __name__ == "__main__":
    main()
