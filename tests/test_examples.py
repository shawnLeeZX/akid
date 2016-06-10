from akid.tests.test import TestCase, main
from akid.examples.mnist import mnist_tf_tutorial


class TestExamples(TestCase):
    def test_mnist_tf_tutorial(self):
        kid = mnist_tf_tutorial.setup()
        precision = kid.practice()
        assert precision >= 0.99


if __name__ == "__main__":
    main()
