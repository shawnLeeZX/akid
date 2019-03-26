from __future__ import absolute_import
from akid import AKID_DATA_PATH
from akid.core import kids, kongfus
from akid import Cifar10TFSource
from akid import IntegratedSensor
from akid import WhitenJoker

from akid.models.brains import VGGNet

# Set up brain
# #########################################################################
brain = VGGNet(padding="SAME",
               name='torch-vgg-cifar10')

# Set up a sensor.
# #########################################################################
cifar_source = Cifar10TFSource(
    name="CIFAR10",
    url='http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
    work_dir=AKID_DATA_PATH + '/cifar10',
    num_train=50000,
    num_val=10000)


sensor = IntegratedSensor(source_in=cifar_source,
                          batch_size=128,
                          name='data')
sensor.attach(WhitenJoker(name="per_image_whitening"))
sensor.attach(WhitenJoker(name="per_image_whitening"), to_val=True)

# Summon a survivor.
# #########################################################################
survivor = kids.Kid(
    sensor,
    brain,
    kongfus.MomentumKongFu(base_lr=1,
                           decay_rate=0.1,
                           decay_epoch_num=25),
    max_steps=510000)
survivor.setup()

# Start training
# #######################################################################
survivor.practice()
