from akid import (
    Kid,
    FeedSensor,
    MomentumKongFu
)

from akid.tests.test import TestCase, TestFactory, main


class TestKongFu(TestCase):
    def test_placeholder_lr_scheme(self):
        from akid import LearningRateScheme
        brain = TestFactory.get_test_brain()
        source = TestFactory.get_test_feed_source()
        kid = Kid(
            FeedSensor(source_in=source, name='data'),
            brain,
            MomentumKongFu(lr_scheme={"name": LearningRateScheme.placeholder}),
            max_steps=900)

        def update_lr(kid):
            if kid.step < 200:
                kid.kongfu.lr_value = 0.1
            elif kid.step < 400:
                kid.kongfu.lr_value = 0.01
            elif kid.step < 600:
                kid.kongfu.lr_value = 0.001
            else:
                kid.kongfu.lr_value = 0.0001

        kid.hooks.on_batch_begin.append(update_lr)
        kid.setup()

        kid.practice()
        assert kid.kongfu.lr_value == 0.0001



if __name__ == "__main__":
    main()
