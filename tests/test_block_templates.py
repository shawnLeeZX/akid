from akid.tests.test import TestCase, main
from akid.examples.mnist import block_mnist


class TestTemplate(TestCase):
    def test_cnn_block(self):
        kid = block_mnist.setup()
        precision = kid.practice()
        assert precision >= 0.98

    def test_bn(self):
        kid = block_mnist.setup(bn={"gamma_init": 1})
        precision = kid.practice()
        assert precision >= 0.98

    def test_activation_before_pooling(self):
        kid = block_mnist.setup(bn={"gamma_init": 1}, activation_before_pooling=True)
        precision = kid.practice()
        assert precision >= 0.98

if __name__ == "__main__":
    main()
