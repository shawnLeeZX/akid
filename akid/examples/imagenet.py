from __future__ import absolute_import
import tensorflow as tf

from akid import AKID_DATA_PATH
from akid.core import kids, kongfus
from akid import ImagenetTFSource
from akid import IntegratedSensor
from akid.models.brains import ImagenetResNet

brain = ImagenetResNet(depth=50, width=2, class_num=1001,
                       name='WRN-50-2')

# Set up a sensor.
# #########################################################################
source = ImagenetTFSource(
    has_super_label=False,
    name="Imagenet",
    url=None,
    work_dir=AKID_DATA_PATH + "/imagenet",
    num_train=1281167,
    num_val=50000)

from akid.core.jokers import (
    Joker,
    CropJoker,
    FlipJoker,
    ResizeJoker
)

sensor = IntegratedSensor(source_in=source,
                          min_fraction_of_examples_in_queue=0.01,
                          num_preprocess_threads=4 * 8,
                          batch_size=256,
                          val_batch_size=200,
                          name='data')
# resize joker
class ColorJoker(Joker):
    def _forward(self, data_in):
      data = tf.image.random_brightness(data_in, max_delta=32. / 255.)
      data = tf.image.random_saturation(data, lower=0.5, upper=1.5)
      data = tf.image.random_hue(data, max_delta=0.2)
      data = tf.image.random_contrast(data, lower=0.5, upper=1.5)
      data = tf.clip_by_value(data, 0.0, 1.0)
      self._data = data


sensor.attach(ResizeJoker(224, 224, name="resize"))
sensor.attach(FlipJoker(flip_left_right=True, name="left_right_flip"))
sensor.attach(ColorJoker(name="color_distortion"))

sensor.attach(
    CropJoker(center=True, central_fraction=0.875, name="center_crop"),
    to_val=True)
sensor.attach(ResizeJoker(224, 224, name="resize"), to_val=True)



class AdhocJoker(Joker):
    def _forward(self, data_in):
        # Make the range of image be [-1, 1]
        data = tf.subtract(data_in, 0.5)
        data = tf.multiply(data, 2.0)
        data = tf.cast(data, tf.float32)
        # because of jpeg decoding, the shape is determined dynamically, thus
        # we manually set shape here.
        data = tf.reshape(data, shape=[224, 224, 3])
        self._data = data


sensor.attach(AdhocJoker(name="to_1_-1"))
sensor.attach(AdhocJoker(name="to_1_-1"), to_val=True)

def update_lr(kid):
    if kid.epoch < 30:
        kid.kongfu.lr_value = 0.1
    elif kid.epoch < 60:
        kid.kongfu.lr_value = 0.01
    else:
        kid.kongfu.lr_value = 0.001
# Summon a survivor.
# #########################################################################
from akid import LearningRateScheme
kid = kids.Kid(
    sensor,
    brain,
    kongfus.MomentumKongFu(lr_scheme={"name": LearningRateScheme.placeholder}),
    engine={"name": "data_parallel", "num_gpu": 8},
    # engine={"name": "single"},
    log_dir="log",
    max_epoch=100)
kid.hooks.on_batch_begin.append(update_lr)
kid.setup()
kid.practice()
