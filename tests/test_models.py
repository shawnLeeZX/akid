from akid.utils.test import AKidTestCase, main, TestFactory


class TestModel(AKidTestCase):
    def test_resnet(self):
        from akid.models import CifarResNet
        # Test ResNet with MNIST.
        brain = CifarResNet(color_channel_num=1,
                            depth=16,
                            width=1,
                            pool_size=7,
                            name="resnet")
        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_kid(source, brain)
        kid.setup()
        loss = kid.practice()
        assert loss < 1.5



if __name__ == "__main__":
    main()
