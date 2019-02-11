from __future__ import print_function

from akid.utils.test import AKidTestCase, TestFactory, main, skipUnless, skip, debug_on
from akid import (
    IntegratedSensor,
    FeedSensor,
    Kid,
    GradientDescentKongFu,
    MomentumKongFu
)
from akid.core.jokers import (
    CropJoker,
    WhitenJoker,
    FlipJoker,
    LightJoker
)

from akid.models.brains import AlexNet
from akid import LearningRateScheme
from akid import backend as A

from akid import SimpleSensor, MNISTSource
import akid

import time


class TestSimpleSensor(AKidTestCase):
    def setUp(self):
        A.reset()
        self.use_cuda_save = A.use_cuda()
        A.use_cuda(False)

    def tearDown(self):
        A.use_cuda(self.use_cuda_save)

    @skipUnless(A.backend() == A.TORCH, msg="Currently MNISTSource depends on torch")
    def test_core(self):
        source = MNISTSource(work_dir="data", name="source")
        source.setup()

        b_size = 32
        sensor = SimpleSensor(source_in=source,
                              batch_size=b_size,
                              queue_size=2,
                              sampler="sequence",
                              name="sensor")
        sensor.setup()

        time.sleep(2)
        d = sensor.forward()
        d_ref = source.get(range(b_size))

        for t in zip(A.eval(d), A.eval(d_ref)):
            self.assertNdarrayEquals(t[0], t[1])

    @skipUnless(A.backend() == A.TORCH, msg="Currently MNISTSource depends on torch")
    def test_summary(self):
        source = MNISTSource(work_dir="data", name="source")
        source.setup()

        b_size = 32
        sensor = SimpleSensor(source_in=source,
                              batch_size=b_size,
                              queue_size=2,
                              sampler="sequence",
                              do_summary_on_val=True,
                              name="sensor")
        sensor.setup()

        sensor.forward()

        sensor.set_mode("val")
        sensor.setup()
        sensor.forward()

        A.summary.init("log_TestSimpleSensor_test_summary")
        summary_ops = A.summary.get_collection(akid.TRAIN_SUMMARY_COLLECTION)
        val_summary_ops = A.summary.get_collection(akid.common.VALID_SUMMARY_COLLECTION)
        summary_ops.extend(val_summary_ops)
        summary_op = A.summary.merge(summary_ops)
        A.summary.run_summary_op(summary_op)

    @skipUnless(A.backend() == A.TORCH, msg="Currently MNISTSource depends on torch")
    def test_do_summary_on_val_flag(self):
        print(A.use_cuda())
        source = MNISTSource(work_dir="data", name="source")
        source.setup()

        b_size = 32
        sensor = SimpleSensor(source_in=source,
                              batch_size=b_size,
                              queue_size=2,
                              sampler="sequence",
                              do_summary_on_val=False,
                              name="sensor")
        sensor.setup()

        sensor.forward()

        sensor.set_mode("val")
        sensor.setup()
        sensor.forward()

        val_summary_ops = A.summary.get_collection(akid.common.VALID_SUMMARY_COLLECTION)
        self.assertEquals(len(val_summary_ops), 0)

        sensor = SimpleSensor(source_in=source,
                              batch_size=b_size,
                              queue_size=2,
                              sampler="sequence",
                              do_summary_on_val=True,
                              name="sensor")
        sensor.set_mode("val")
        sensor.setup()
        sensor.forward()

        val_summary_ops = A.summary.get_collection(akid.common.VALID_SUMMARY_COLLECTION)
        self.assertNotEquals(len(val_summary_ops), 0)

class TestFeedSensor(AKidTestCase):
    def setUp(self):
        super(TestFeedSensor, self).setUp()
        A.reset()
        self.brain = TestFactory.get_test_brain()
        source = TestFactory.get_test_feed_source()
        self.sensor = FeedSensor(source_in=source,
                                 batch_size=128,
                                 val_batch_size=100,
                                 name="data")

    @skipUnless(A.backend() == A.TF)
    def test_core(self):
        """
        Test core functionality of feed sensor. More specifically, the
        different between the core test in `TestKid` and this test is a
        different validation batch size with training batch size, which needs
        more logic to handle.
        """
        kid = Kid(
            self.sensor,
            self.brain,
            MomentumKongFu(),
            max_steps=900)
        kid.setup()
        loss = kid.practice()

        assert loss < 0.2

    @skipUnless(A.backend() == A.TF)
    def test_summary_on_val(self):
        """
        Test whether validation summaries has been written to event file
        properly. Besides proper execution, whether summaries have been written
        to event files properly needs manual check by launching tensorboard. It
        may be upgraded to use the tensorflow read even file functionality in
        the future.
        """
        kid = Kid(
            self.sensor,
            self.brain,
            MomentumKongFu(),
            max_steps=900,
            do_summary_on_val=True)
        kid.setup()
        kid.practice()


class TestIntegratedSensor(AKidTestCase):
    def setUp(self):
        super(TestIntegratedSensor, self).setUp()
        A.reset()

        # TODO(Shuai): This test is supposed to test on MNIST with
        # integrated sensor instead of using data augmented cifar10.
        self.brain = AlexNet(in_channel_num=2304, dataset="cifar10", name="AlexNet")
        source = TestFactory.get_test_tf_source()

        sensor = IntegratedSensor(source_in=source,
                                  batch_size=128,
                                  val_batch_size=100,
                                  name='data')
        sensor.attach(CropJoker(height=24, width=24,
                                center=True, name="crop"),
                      to_val=True)
        sensor.attach(WhitenJoker(name="per_image_whitening"), to_val=True)

        sensor.attach(CropJoker(height=24, width=24, name="crop"))
        sensor.attach(FlipJoker(name="left_right_flip"))
        sensor.attach(LightJoker(name="brightness_contrast"))
        sensor.attach(WhitenJoker(name="per_image_whitening"))

        self.sensor = sensor

    @skip("Cifar10 test fails for now.")
    # @skipUnless(A.backend() == A.TF)
    def test_core(self):
        kid = Kid(
            self.sensor,
            self.brain,
            GradientDescentKongFu(
                lr_scheme={"name": LearningRateScheme.exp_decay,
                           "base_lr": 0.1,
                           "decay_rate": 0.1,
                           "num_batches_per_epoch": 391,
                           "decay_epoch_num": 350}),
            max_steps=1000)
        kid.setup()

        loss = kid.practice()
        assert loss < 3.4

    @skip("Cifar10 test fails for now.")
    # @skipUnless(A.backend() == A.TF)
    def test_summary_on_val(self):
        kid = Kid(
            self.sensor,
            self.brain,
            GradientDescentKongFu(
                lr_scheme={"name": LearningRateScheme.exp_decay,
                           "base_lr": 0.1,
                           "decay_rate": 0.1,
                           "num_batches_per_epoch": 391,
                           "decay_epoch_num": 350}),
            max_steps=200,
            do_summary_on_val=True)
        kid.setup()

        kid.practice()

if __name__ == "__main__":
    main()
