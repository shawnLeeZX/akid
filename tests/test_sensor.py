from akid.tests.test import TestCase, TestFactory, main
from akid import (
    IntegratedSensor,
    FeedSensor,
    Survivor,
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


class TestFeedSensor(TestCase):
    def setUp(self):
        self.brain = TestFactory.get_test_brain()
        source = TestFactory.get_test_feed_source()
        self.sensor = FeedSensor(source_in=source,
                                 batch_size=128,
                                 val_batch_size=100,
                                 name="data")

    def test_core(self):
        """
        Test core functionality of feed sensor. More specifically, the
        different between the core test in TestSurvivor and this test is a
        different validation batch size with training batch size, which needs
        more logic to handle.
        """
        kid = Survivor(
            self.sensor,
            self.brain,
            MomentumKongFu(),
            max_steps=900)
        kid.setup()
        loss = kid.practice()

        assert loss < 0.2

    def test_summary_on_val(self):
        """
        Test whether validation summaries has been written to event file
        properly. Besides proper execution, whether summaries have been written
        to event files properly needs manual check by launching tensorboard. It
        may be upgraded to use the tensorflow read even file functionality in
        the future.
        """
        kid = Survivor(
            self.sensor,
            self.brain,
            MomentumKongFu(),
            max_steps=900,
            summary_on_val=True)
        kid.setup()
        kid.practice()


class TestIntegratedSensor(TestCase):
    def setUp(self):
        # TODO(Shuai): This test is supposed to test on MNIST with
        # integrated sensor instead of using data augmented cifar10.
        self.brain = AlexNet(name="AlexNet")
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

    def test_core(self):
        kid = Survivor(
            self.sensor,
            self.brain,
            GradientDescentKongFu(base_lr=0.1,
                                  decay_rate=0.1,
                                  decay_epoch_num=350),
            max_steps=1000)
        kid.setup()

        loss = kid.practice()
        assert loss < 3.4

    def test_summary_on_val(self):
        kid = Survivor(
            self.sensor,
            self.brain,
            GradientDescentKongFu(base_lr=0.1,
                                  decay_rate=0.1,
                                  decay_epoch_num=350),
            max_steps=200,
            summary_on_val=True)
        kid.setup()

        kid.practice()

if __name__ == "__main__":
    main()
