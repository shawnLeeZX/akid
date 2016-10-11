from akid.utils.test import AKidTestCase, TestFactory, main
from akid import (
    IntegratedSensor,
    RescaleJoker,
    Kid,
    GradientDescentKongFu
)
from akid.models.brains import AlexNet
from akid import LearningRateScheme


class TestJoker(AKidTestCase):
    def test_rescale_joker(self):
        # TODO(Shuai): This test is supposed to test on MNIST with integrated
        # sensor instead of using cifar10.
        brain = AlexNet(name="AlexNet")
        source = TestFactory.get_test_tf_source()

        sensor = IntegratedSensor(source_in=source,
                                  batch_size=128,
                                  val_batch_size=100,
                                  name='data')
        sensor.attach(RescaleJoker(name="rescale"), to_val=True)

        sensor.attach(RescaleJoker(name="rescale"))

        kid = Kid(
            sensor,
            brain,
            GradientDescentKongFu(
                lr_scheme={
                    "name": LearningRateScheme.exp_decay,
                    "base_lr": 0.1,
                    "decay_rate": 0.1,
                    "num_batches_per_epoch": 391,
                    "decay_epoch_num": 350}),
            max_steps=2000)

        kid.setup()

        loss = kid.practice()
        assert loss < 3


if __name__ == "__main__":
    main()
