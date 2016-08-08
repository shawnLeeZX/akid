from akid import (
    Kid,
    FeedSensor,
    MomentumKongFu,
    MNISTFeedSource,
    AKID_DATA_PATH
)
from akid.models import LeNet

from akid.tests.test import TestCase, TestFactory, main


class TestEngine(TestCase):
    def test_data_parallel(self):
        brain = LeNet(name="LeNet")
        source = MNISTFeedSource(name="MNIST",
                                 url='http://yann.lecun.com/exdb/mnist/',
                                 work_dir=AKID_DATA_PATH + '/mnist',
                                 center=True,
                                 scale=True,
                                 num_train=50000,
                                 num_val=10000)
        kid = Kid(
            FeedSensor(source_in=source, name='data'),
            brain,
            MomentumKongFu(name="opt"),
            engine="data_parallel",
            max_steps=1000)
        kid.setup()

        loss = kid.practice()
        assert loss < 3


if __name__ == "__main__":
    main()
