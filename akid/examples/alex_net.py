from __future__ import absolute_import
def setup():
    from akid import AKID_DATA_PATH
    from akid import Kid
    from akid import LearningRateScheme, GradientDescentKongFu
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
    brain = AlexNet(in_channel_num=2304, dataset="cifar10", name='alex_net')

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

    sensor.attach(CropJoker(height=24, width=24, center=False, name="crop"))
    sensor.attach(FlipJoker(name="left_right_flip"))
    sensor.attach(LightJoker(name="brightness_contrast"))
    sensor.attach(WhitenJoker(name="per_image_whitening"))

    # Summon a survivor.
    # #########################################################################
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
        max_steps=510000)
    kid.setup()

    return kid

# Start training
# #######################################################################
if __name__ == "__main__":
    kid = setup()
    kid.practice(continue_from_chk_point=False)
