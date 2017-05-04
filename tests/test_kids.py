import os

from akid import (
    Kid,
    FeedSensor,
    MomentumKongFu
)

from akid.utils.test import AKidTestCase, TestFactory, main


class TestKid(AKidTestCase):
    def test_core(self):
        brain = TestFactory.get_test_brain()
        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_kid(source, brain)
        kid.setup()

        loss = kid.practice()
        assert loss < 0.2, \
                "Loss: {}".format(loss)

    def test_saver(self):
        brain = TestFactory.get_test_brain()
        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_kid(source, brain)
        kid.setup()

        loss = kid.practice()
        assert loss < 0.2

        kid.restore_from_ckpt()
        loss, _ = kid.validate()
        assert loss < 0.2, \
                "Loss is {}".format(loss)

    def test_log_to_file_flag(self):
        brain = TestFactory.get_test_brain()
        source = TestFactory.get_test_feed_source()
        kid = Kid(
            FeedSensor(source_in=source, name='data'),
            brain,
            MomentumKongFu(),
            log_dir="log_test_kid",
            log_to_file=False,
            max_steps=900)
        kid.setup()

        assert not os.path.exists(kid.log_dir + "/training.log)")


if __name__ == "__main__":
    main()
