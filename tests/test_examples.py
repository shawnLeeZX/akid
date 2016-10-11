from akid.utils.test import AKidTestCase, main
from akid.examples.mnist import mnist_tf_tutorial


class TestExamples(AKidTestCase):
    def test_mnist_tf_tutorial(self):
        kid = mnist_tf_tutorial.setup()
        loss = kid.practice()
        assert loss < 2.4


if __name__ == "__main__":
    main()
