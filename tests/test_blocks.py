import tensorflow as tf

from akid.tests.test import TestCase, main, TestFactory
from akid import Brain, FeedSensor, MomentumKongFu, Survivor
from akid.layers import (
    ConvolutionLayer,
    PoolingLayer,
    ReLULayer,
    InnerProductLayer,
    SoftmaxWithLossLayer,
)


class TestBlock(TestCase):
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

        brain.attach(SoftmaxWithLossLayer(class_num=10, name="loss"))

        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_survivor(source, brain)
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
        brain.attach(SoftmaxWithLossLayer(class_num=10, name="loss"))

        source = TestFactory.get_test_feed_source()

        graph = tf.Graph()
        kid = Survivor(
            FeedSensor(source_in=source, name='data'),
            brain,
            MomentumKongFu(),
            log_dir="log",
            max_steps=900,
            graph=graph)
        kid.setup()
        W = kid.brain.get_filters()[0]
        with tf.Session(graph=graph) as sess:
            kid.practice(sess)
            W_norm = sess.run(tf.global_norm([W[:, :, :, 0]]))
            print(W_norm)
            assert W_norm <= 1

if __name__ == "__main__":
    main()
