import os

import tensorflow as tf

from akid import (
    Survivor,
    FeedSensor,
    MomentumKongFu
)

from akid.tests.test import TestCase, TestFactory, main


class TestSurvivor(TestCase):
    def test_core(self):
        brain = TestFactory.get_test_brain()
        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_survivor(source, brain)
        kid.setup()

        loss = kid.practice()
        assert loss < 0.2

    def test_saver(self):
        brain = TestFactory.get_test_brain()
        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_survivor(source, brain)
        kid.setup()

        sess = tf.Session(
            graph=kid.graph,
            config=tf.ConfigProto(allow_soft_placement=True))
        with sess:
            loss = kid.practice()
            assert loss < 0.2

            kid.restore_from_ckpt(sess)
            loss = kid.validate(sess)
            assert loss < 0.2

    def test_log_to_file_flag(self):
        brain = TestFactory.get_test_brain()
        source = TestFactory.get_test_feed_source()
        kid = Survivor(
            FeedSensor(source_in=source, name='data'),
            brain,
            MomentumKongFu(),
            log_dir="log_test_survivor",
            log_to_file=False,
            max_steps=900)
        kid.setup()

        assert not os.path.exists(kid.log_dir + "/training.log)")


if __name__ == "__main__":
    main()
