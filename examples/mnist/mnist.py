from akid import AKID_DATA_PATH
from akid import FeedSensor
from akid import Kid
from akid import MomentumKongFu
from akid import MNISTFeedSource

from akid.models.brains import LeNet

brain = LeNet(name="LeNet")
source = MNISTFeedSource(name="MNIST",
                         url='http://yann.lecun.com/exdb/mnist/',
                         work_dir=AKID_DATA_PATH + '/mnist',
                         center=True,
                         scale=True,
                         num_train=50000,
                         num_val=10000)

s = Kid(FeedSensor(name='data', source_in=source),
             brain,
             MomentumKongFu())
s.setup()

s.practice()
