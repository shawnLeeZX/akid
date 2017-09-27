from akid import AKID_DATA_PATH
from akid import FeedSensor
from akid import Kid
from akid import MomentumKongFu
from akid import MNISTFeedSource

from akid.models.brains import NewMnistTfTutorialNet
from akid import LearningRateScheme
from akid.utils.test import TestFactory
from akid import backend as A


def setup():
    brain = NewMnistTfTutorialNet(name="mnist-tf-tutorial-net")
    sensor = TestFactory.get_test_sensor()
    kid = Kid(
        sensor,
        brain,
        MomentumKongFu(
            # lr_scheme here does not have effect, only left for reference.
            lr_scheme={
                "name": LearningRateScheme.exp_decay,
                "base_lr": 0.01,
                "decay_rate": 0.95,
                "num_batches_per_epoch": 468,
                "decay_epoch_num": 1},
            momentum=0.9),
        max_steps=20000)
    kid.setup()

    return kid


if __name__ == "__main__":
    kid = setup()
    kid.practice()
