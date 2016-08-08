from akid import AKID_DATA_PATH
from akid import FeedSensor
from akid import Kid
from akid import MomentumKongFu
from akid import MNISTFeedSource

from akid.models.brains import MnistTfTutorialNet


def setup():
    brain = MnistTfTutorialNet(name="mnist-tf-tutorial-net")
    source = MNISTFeedSource(name="MNIST",
                             url='http://yann.lecun.com/exdb/mnist/',
                             work_dir=AKID_DATA_PATH + '/mnist',
                             num_train=50000,
                             num_val=5000,
                             center=True,
                             scale=True)

    kid = Kid(FeedSensor(name='data',
                              source_in=source,
                              batch_size=64,
                              val_batch_size=100),
                   brain,
                   MomentumKongFu(momentum=0.9,
                                  base_lr=0.01,
                                  decay_rate=0.95,
                                  decay_epoch_num=1),
                   max_steps=4000)
    kid.setup()

    return kid


if __name__ == "__main__":
    kid = setup()
    kid.practice()
