from __future__ import absolute_import
import os
import numpy as np

from akid import (
    Kid,
    FeedSensor,
    MomentumKongFu
)

from akid.utils.test import AKidTestCase, TestFactory, main, debug_on
from akid import backend as A
from akid.core import initializers

class TestKid(AKidTestCase):
    def test_core(self):
        brain = TestFactory.get_test_brain()
        sensor = TestFactory.get_test_sensor()
        kid = TestFactory.get_test_kid(sensor, brain)
        kid.do_summary = False
        kid.setup()

        loss = kid.practice()
        assert loss < 0.2, \
                "Loss: {}".format(loss)

        kid.teardown()

    def test_verbose_eval_blocks(self):
        brain = TestFactory.get_test_brain()
        l = brain.get_layer_by_name("loss")
        def f(self, *args, **kwargs):
            self._verbose_eval = self.eval
        l.post_forward_hook.append(f)
        sensor = TestFactory.get_test_sensor()
        kid = TestFactory.get_test_kid(sensor, brain)
        kid.do_summary = False
        kid.setup()

        loss = kid.practice()
        assert loss < 0.2, \
                "Loss: {}".format(loss)
        kid.teardown()

    def test_summary(self):
        brain = TestFactory.get_test_brain()
        sensor = TestFactory.get_test_sensor()
        brain.set_flag("summarize_variables", True)
        brain.set_flag("summarize_output", True)
        kid = TestFactory.get_test_kid(sensor, brain)
        kid.do_summary = True
        kid.setup()

        loss = kid.practice()
        assert loss < 0.2, \
                "Loss: {}".format(loss)
        kid.teardown()

    def test_summary_on_val(self):
        brain = TestFactory.get_test_brain()
        sensor = TestFactory.get_test_sensor()
        brain.set_flag("summarize_variables", True)
        brain.set_flag("summarize_output", True)
        kid = TestFactory.get_test_kid(sensor, brain)
        kid.do_summary = True
        kid.do_summary_on_val = True
        kid.setup()

        loss = kid.practice()
        assert loss < 0.2, \
                "Loss: {}".format(loss)
        kid.teardown()

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
        self.assertEquals(A.get_step(), 900)

        kid.teardown()

    def test_log_to_file_flag(self):
        brain = TestFactory.get_test_brain()
        kid = Kid(
            TestFactory.get_test_sensor(),
            brain,
            MomentumKongFu(),
            log_dir="log_test_kid",
            log_to_file=False,
            max_steps=900)
        kid.setup()

        assert not os.path.exists(kid.log_dir + "/training.log)")

        kid.teardown()

    def test_batch_monitoring(self):
        brain = TestFactory.get_test_brain()
        sensor = TestFactory.get_test_sensor()
        kid = TestFactory.get_test_kid(sensor, brain)
        kid.do_summary = True
        kid.do_batch_monitoring = True
        kid.train_log_step = 10

        # The last layer is the loss layer, add a hook to it to do monitoring.
        from akid import BATCH_MONITORING_COLLECTION
        l = kid.brain.blocks[-1]
        def summarize_loss(self, *args, **kwargs):
            if self.is_mon and not self.done_first_batch_monitoring_pass:
                A.summary.scalar(A.get_name(self.loss), self.loss, collections=[BATCH_MONITORING_COLLECTION])
        l.post_forward_hook.append(summarize_loss)

        kid.setup()

        kid.init()
        kid.on_train_begin()
        # Reset sensor, so to prevent any code from using some training batches
        # in advance, and making them missing from initial training.
        kid.sensor.reset()

        cached_data = A.eval(kid.cached_data)
        # Test that the same batch is being monitored throughout
        def batch_checking(kid, *args, **kwargs):
            self.assertNdarrayEquals(A.eval(kid.cached_data), cached_data)

        for i in range(100):
            val_loss, val_evals = kid.step_with_logistics()

        kid.teardown()

if __name__ == "__main__":
    main()
