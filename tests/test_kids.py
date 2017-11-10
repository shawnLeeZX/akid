import os

from akid import (
    Kid,
    FeedSensor,
    MomentumKongFu
)

from akid.utils.test import AKidTestCase, TestFactory, main, debug_on
from akid import backend as A


class TestKid(AKidTestCase):
    def setUp(self):
        A.reset()

    def test_core(self):
        brain = TestFactory.get_test_brain()
        sensor = TestFactory.get_test_sensor()
        kid = TestFactory.get_test_kid(sensor, brain)
        kid.do_summary = False
        kid.setup()

        loss = kid.practice()
        assert loss < 0.2, \
                "Loss: {}".format(loss)

    def test_summary(self):
        brain = TestFactory.get_test_brain()
        sensor = TestFactory.get_test_sensor()
        kid = TestFactory.get_test_kid(sensor, brain)
        kid.do_summary = True
        kid.setup()

        loss = kid.practice()
        assert loss < 0.2, \
                "Loss: {}".format(loss)

    def test_summary_on_val(self):
        brain = TestFactory.get_test_brain()
        sensor = TestFactory.get_test_sensor()
        kid = TestFactory.get_test_kid(sensor, brain)
        kid.do_summary = True
        kid.do_summary_on_val = True
        kid.setup()

        loss = kid.practice()
        assert loss < 0.2, \
                "Loss: {}".format(loss)

    def test_saver(self):
        lr_ref = 0.001
        def update_lr(kid):
            if A.get_step() > 890:
                kid.kongfu.set_lr(lr_ref)

        brain = TestFactory.get_test_brain()
        sensor = TestFactory.get_test_sensor()
        kid = TestFactory.get_test_kid(sensor, brain)
        kid.max_steps = 900
        kid.hooks.on_batch_begin.append(update_lr)
        kid.setup()

        loss = kid.practice()
        self.assertLess(loss, 0.2)

        A.get_variable_scope().reuse_variables()
        kid.continue_from_chk_point = True
        kid.inference_mode = True
        kid.setup()
        self.assertEquals(kid.kongfu.get_lr(), lr_ref)
        loss_recovered, _ = kid.validate()

        # WARNING: the recovered loss is not exactly the same for tensorflow
        # backend. For now it is classified as numerical reason, but it should
        # be further investigated when using tensorflow.
        self.assertLess(abs(loss - loss_recovered), 10e-3)
        # The extra 1 is caused by the breaking of the loop
        self.assertEquals(A.get_step(), 901)

    # def test_log_to_file_flag(self):
    #     brain = TestFactory.get_test_brain()
    #     source = TestFactory.get_test_feed_source()
    #     kid = Kid(
    #         FeedSensor(source_in=source, name='data'),
    #         brain,
    #         MomentumKongFu(),
    #         log_dir="log_test_kid",
    #         log_to_file=False,
    #         max_steps=900)
    #     kid.setup()

    #     assert not os.path.exists(kid.log_dir + "/training.log)")


if __name__ == "__main__":
    main()
