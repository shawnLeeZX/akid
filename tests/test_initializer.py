import numpy as np

from akid.utils.test import AKidTestCase, main, TestFactory
from akid import initializers
from akid import GraphBrain
from akid.sugar import cnn_block
import akid.backend as A
from akid.layers import SoftmaxWithLossLayer


class TestInitializer(AKidTestCase):
    def test_tensor(self):
        X_np = np.array([[1, 1], [2, 3]])
        init = initializers.get("tensor", value=X_np)
        X = A.get_variable("X", initializer=init)
        A.init()
        X_eval = A.eval(X)
        assert (X_eval == X_np).all()

    def test_uniform_unit_scale_initializer(self):
        brain = GraphBrain(name="test_brain")
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

    def test_unit_gradient_initializer(self):
        brain = GraphBrain(name="test_brain")
        brain.attach(cnn_block(ksize=[5, 5],
                               initial_bias_value=0.,
                               init_para={"name": "msra"},
                               wd={"type": "l2", "scale": 0.0005},
                               out_channel_num=32,
                               pool_size=[5, 5],
                               pool_stride=[5, 5],
                               activation={"type": "relu"}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.,
                               init_para={"name": "msra"},
                               wd={"type": "l2", "scale": 0.0005},
                               out_channel_num=512,
                               activation={"type": "relu"}))

        brain.attach(cnn_block(ksize=None,
                               initial_bias_value=0.,
                               init_para={"name": "msra"},
                               wd={"type": "l2", "scale": 0.0005},
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
