from akid import AKID_DATA_PATH
from akid.core import kids, kongfus
from akid import Cifar10TFSource
from akid import IntegratedSensor
from akid import (
    CropJoker,
    WhitenJoker,
    FlipJoker,
    LightJoker
)

from akid.models.brains import AlexNet

# Set up brain
# #########################################################################
brain = AlexNet(moving_average_decay=0.99, name='alex-google-cifar10')

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
sensor.attach(CropJoker(height=24, width=24, center=True, name="crop"),
              to_val=True)
sensor.attach(WhitenJoker(name="per_image_whitening"), to_val=True)

sensor.attach(CropJoker(height=24, width=24, center=True, name="crop"))
sensor.attach(FlipJoker(name="left_right_flip"))
sensor.attach(LightJoker(name="brightness_contrast"))
sensor.attach(WhitenJoker(name="per_image_whitening"))

# Summon a survivor.
# #########################################################################
survivor = kids.Kid(
    sensor,
    brain,
    kongfus.GradientDescentKongFu(base_lr=0.1,
                                  decay_rate=0.1,
                                  decay_epoch_num=350),
    max_steps=510000)
survivor.setup()

# Start training
# #######################################################################
survivor.practice(continue_from_chk_point=False)
