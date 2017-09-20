import tensorflow as tf
import numpy as np

from akid.utils.test import AKidTestCase, main, TestFactory
from akid import GraphBrain
from akid.sugar import cnn_block
from akid import sugar
from akid.layers import SoftmaxWithLossLayer
from akid import backend as A


class TestLossLayers(AKidTestCase):
    def setUp(self):
        super(TestLossLayers, self).setUp()
        sugar.init()

    # def test_dense_eval(self):
    #     labels = tf.constant([[1, 0, 0], [0, 1, 0]], dtype=tf.float32)
    #     logits = tf.constant([[1, 2, 0], [1, 2, 0]], dtype=tf.float32)
    #     l = SoftmaxWithLossLayer(class_num=3, name="loss")
    #     l.forward([logits, labels])
    #     with tf.Session():
    #         assert l.eval.eval() == 0.5

    def test_mse_loss(self):
        from akid.layers import MSELossLayer

        X_in = A.Tensor([[1, 1], [0, 0]])
        label_in = A.Tensor([[1, 0], [0, 0]])
        l = MSELossLayer(name='loss')
        X_out = l.forward([X_in, label_in])
        X_out_ref = 0.25

        A.init()
        X_out_eval = A.eval(X_out)
        self.assertEquals(X_out_eval, X_out_ref)


    # def test_multiplier(self):
    #     brain = GraphBrain(name="test_brain")
    #     brain.attach(cnn_block(ksize=[5, 5],
    #                            initial_bias_value=0.,
    #                            init_para={"name": "truncated_normal",
    #                                       "stddev": 0.1},
    #                            wd={"type": "l2", "scale": 0.0005},
    #                            in_channel_num=1,
    #                            out_channel_num=32,
    #                            pool_size=[5, 5],
    #                            pool_stride=[5, 5],
    #                            activation={"type": "gsmax",
    #                                        "group_size": 2}))

    #     brain.attach(cnn_block(ksize=None,
    #                            initial_bias_value=0.1,
    #                            init_para={"name": "truncated_normal",
    #                                       "stddev": 0.1},
    #                            wd={"type": "l2", "scale": 0.0005},
    #                            in_channel_num=1152,
    #                            out_channel_num=512,
    #                            activation={"type": "gsmax",
    #                                        "group_size": 8}))

    #     brain.attach(cnn_block(ksize=None,
    #                            initial_bias_value=0.1,
    #                            init_para={"name": "truncated_normal",
    #                                       "stddev": 0.1},
    #                            wd={"type": "l2", "scale": 0.0005},
    #                            in_channel_num=512,
    #                            out_channel_num=10,
    #                            activation=None))
    #     brain.attach(SoftmaxWithLossLayer(
    #         multiplier=0.5,
    #         class_num=10,
    #         inputs=[{"name": "ip3", "idxs": [0]},
    #                 {"name": "system_in", "idxs": [1]}],
    #         name="loss"))

    #     source = TestFactory.get_test_feed_source()
    #     kid = TestFactory.get_test_kid(source, brain)
    #     kid.setup()
    #     kid.init()
    #     kid.fill_train_feed_dict()
    #     with kid.sess:
    #         layer = kid.engine.get_layer(name="loss")
    #         loss = layer.loss.eval(feed_dict=kid.feed_dict)
    #         assert loss < 1.5

if __name__ == "__main__":
    main()
