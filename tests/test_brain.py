from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf

from akid.utils.test import AKidTestCase, TestFactory, main, debug_on, skipUnless
from akid import GraphBrain, FeedSensor, MomentumKongFu, Kid
from akid.layers import (
    ConvolutionLayer,
    PoolingLayer,
    ReLULayer,
    InnerProductLayer,
    SoftmaxWithLossLayer,
)
from akid import backend as A

from akid.utils import glog as log
from six.moves import zip
log.init()


class TestBrain(AKidTestCase):
    def setUp(self):
        self.use_cuda_save = A.use_cuda()
        A.use_cuda(False)
        A.reset()

    def tearDown(self):
        A.use_cuda(self.use_cuda_save)

    def test_forward(self):
        from akid.layers import MSELossLayer, InnerProductLayer
        from akid import GraphBrain

        X_in = A.Tensor([[1., 1], [1, 0]], requires_grad=True)
        weight = A.Tensor([
            [1., 1],
            [0, 1]
        ])
        label_in = A.Tensor([
            [1., 2],
            [1, 1]
        ],
                            requires_grad=True
        )
        label_in = label_in + 1

        X_out_ref = 0
        mse_loss_ref = 0
        wd_loss_ref = 1.5
        loss_ref = mse_loss_ref + wd_loss_ref

        b = GraphBrain(name='test_brain')
        b.attach(
            InnerProductLayer(in_channel_num=2,
                              out_channel_num=2,
                              initial_bias_value=1.,
                              init_para={"name": "tensor",
                                         "value": weight},
                              wd={"type": "l2", "scale": 1},
                              name='ip'))

        b.attach(MSELossLayer(inputs=[{"name": "ip"},
                                      {"name": "system_in", "idxs": [1]}],
                              name='loss'))

        X_out = b.forward([X_in, label_in])

        A.init()

        X_out_eval = A.eval(X_out[0])
        self.assertEquals(X_out_eval, X_out_ref)

        loss_eval = A.eval(b.loss)
        self.assertEquals(loss_eval, loss_ref)

    @skipUnless(A.backend() == A.TF)
    def test_moving_average(self):
        brain = TestFactory.get_test_brain(using_moving_average=True)
        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_kid(source, brain)
        kid.setup()

        loss = kid.practice()
        assert loss < 0.2

    def assert_diff(self, brain_a, brain_b):
        """
        Compare two brains, and make sure they are completely different two.
        """
        for block_a, block_b in zip(brain_a.blocks, brain_b.blocks):
            assert block_a != block_b

    def test_copy(self):
        brain = TestFactory.get_test_brain(using_moving_average=True)
        brain_copy = brain.get_copy()
        self.assert_diff(brain, brain_copy)

    @skipUnless(A.backend() == A.TF)
    def test_val_copy(self):
        brain = TestFactory.get_test_brain(using_moving_average=True)
        val_brain = brain.get_val_copy()
        self.assert_diff(brain, val_brain)
        for b in val_brain.blocks:
            assert b.is_val is True

    @skipUnless(A.backend() == A.TF)
    def test_initialization(self):
        """
        This is to test initialization could be properly set up. It does not
        involve run time things.
        """
        brain = GraphBrain(name="Test")
        brain.attach(
            ConvolutionLayer(ksize=[5, 5],
                             strides=[1, 1, 1, 1],
                             padding="SAME",
                             init_para={"name": "truncated_normal",
                                        "stddev": 0.1},
                             in_channel_num=1,
                             out_channel_num=32,
                             name="conv1")
        )
        brain.attach(ReLULayer(name="relu1"))
        brain.attach(
            PoolingLayer(ksize=[1, 5, 5, 1],
                         strides=[1, 5, 5, 1],
                         padding="SAME",
                         name="pool1")
        )

        brain.attach(InnerProductLayer(in_channel_num=1152, out_channel_num=10, name="ip1"))

        brain.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[{"name": "ip1", "idxs": [0]},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))

        sensor = TestFactory.get_test_sensor()
        kid = TestFactory.get_test_kid(sensor, brain)
        kid.setup()

    @skipUnless(A.backend() == A.TF)
    def test_max_norm(self):
        brain = GraphBrain(name="Test")
        brain.attach(
            ConvolutionLayer(ksize=[5, 5],
                             strides=[1, 1, 1, 1],
                             padding="SAME",
                             init_para={"name": "truncated_normal",
                                        "stddev": 0.1},
                             in_channel_num=1,
                             out_channel_num=32,
                             max_norm=1,
                             # Do not use bias since we only care about the
                             # weights now.
                             initial_bias_value=None,
                             name="conv1")
        )
        brain.attach(InnerProductLayer(in_channel_num=25088, out_channel_num=10, name="ip1"))
        brain.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[{"name": "ip1", "idxs": [0]},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))

        source = TestFactory.get_test_feed_source()

        kid = Kid(
            FeedSensor(source_in=source, name='data'),
            brain,
            MomentumKongFu(),
            log_dir="log",
            max_steps=900)
        kid.setup()
        W = kid.brain.get_filters()[0]
        kid.practice()
        W_norm = A.eval(tf.global_norm([W[:, :, :, 0]]))
        print(W_norm)
        assert W_norm <= 1

if __name__ == "__main__":
    main()
