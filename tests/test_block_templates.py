from akid.utils.test import AKidTestCase, main
from akid.examples.mnist import block_mnist
from akid import sugar


class TestTemplate(AKidTestCase):
    def setUp(self):
        super(TestTemplate, self).setUp()
        sugar.init()

    def test_cnn_block(self):
        kid = block_mnist.setup()
        loss = kid.practice()
        assert loss < 2.5

    def test_activation_before_pooling(self):
        kid = block_mnist.setup(activation_before_pooling=True)
        loss = kid.practice()
        assert loss < 2.5

if __name__ == "__main__":
    main()
