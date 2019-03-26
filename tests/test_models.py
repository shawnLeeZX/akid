from __future__ import absolute_import
from akid.utils.test import AKidTestCase, main, TestFactory, skipUnless

from akid import backend as A


class TestModel(AKidTestCase):
    def setUp(self):
        A.reset()

    @skipUnless(A.backend() == A.TF)
    def test_resnet(self):
        from akid.models import CifarResNet
        # Test ResNet with MNIST.
        brain = CifarResNet(color_channel_num=1,
                            depth=16,
                            width=1,
                            pool_size=7,
                            name="resnet")
        sensor = TestFactory.get_test_sensor()
        kid = TestFactory.get_test_kid(sensor, brain)
        kid.setup()
        loss = kid.practice()
        assert loss < 1.5



if __name__ == "__main__":
    main()
