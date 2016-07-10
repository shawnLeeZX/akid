from akid.tests.test import TestCase, main
from akid.examples.mnist import block_mnist
from akid import sugar


class TestTemplate(TestCase):
    def setUp(self):
        sugar.init()

    def test_cnn_block(self):
        kid = block_mnist.setup()
        loss = kid.practice()
        assert loss < 2.5

    def test_bn(self):
        kid = block_mnist.setup(bn={"gamma_init": 1})
        loss = kid.practice()
        assert loss < 2.5

    def test_activation_before_pooling(self):
        kid = block_mnist.setup(
            bn={"gamma_init": 1}, activation_before_pooling=True)
        loss = kid.practice()
        assert loss < 2.5

if __name__ == "__main__":
    main()
