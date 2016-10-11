import tensorflow as tf

from akid.utils.test import AKidTestCase, TestFactory, main
from akid import Brain, FeedSensor, MomentumKongFu, Kid
from akid.layers import (
    ConvolutionLayer,
    PoolingLayer,
    ReLULayer,
    InnerProductLayer,
    SoftmaxWithLossLayer,
)


class TestBrain(AKidTestCase):
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

    def test_val_copy(self):
        brain = TestFactory.get_test_brain(using_moving_average=True)
        val_brain = brain.get_val_copy()
        self.assert_diff(brain, val_brain)
        for b in val_brain.blocks:
            assert b.is_val is True

    def test_initialization(self):
        """
        This is to test initialization could be properly set up. It does not
        involve run time things.
        """
        brain = Brain(name="Test")
        brain.attach(
            ConvolutionLayer(ksize=[5, 5],
                             strides=[1, 1, 1, 1],
                             padding="SAME",
                             init_para={"name": "truncated_normal",
                                        "stddev": 0.1},
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

        brain.attach(InnerProductLayer(out_channel_num=10, name="ip1"))

        brain.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[{"name": "ip1", "idxs": [0]},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))

        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_kid(source, brain)
        kid.setup()

    def test_max_norm(self):
        brain = Brain(name="Test")
        brain.attach(
            ConvolutionLayer(ksize=[5, 5],
                             strides=[1, 1, 1, 1],
                             padding="SAME",
                             init_para={"name": "truncated_normal",
                                        "stddev": 0.1},
                             out_channel_num=32,
                             max_norm=1,
                             # Do not use bias since we only care about the
                             # weights now.
                             initial_bias_value=None,
                             name="conv1")
        )
        brain.attach(InnerProductLayer(out_channel_num=10, name="ip1"))
        brain.attach(SoftmaxWithLossLayer(
            class_num=10,
            inputs=[{"name": "ip1", "idxs": [0]},
                    {"name": "system_in", "idxs": [1]}],
            name="loss"))

        source = TestFactory.get_test_feed_source()

        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            kid = Kid(
                FeedSensor(source_in=source, name='data'),
                brain,
                MomentumKongFu(),
                sess=sess,
                log_dir="log",
                max_steps=900,
                graph=graph)
            kid.setup()
            W = kid.brain.get_filters()[0]
            kid.practice()
            W_norm = kid.sess.run(tf.global_norm([W[:, :, :, 0]]))
            print(W_norm)
            assert W_norm <= 1

if __name__ == "__main__":
    main()
