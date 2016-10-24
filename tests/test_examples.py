from akid.utils.test import AKidTestCase, main


class TestExamples(AKidTestCase):
    def test_mnist_tf_tutorial(self):
        from akid.examples.mnist import mnist_tf_tutorial
        kid = mnist_tf_tutorial.setup()
        loss = kid.practice()
        assert loss < 2.4

    def test_alex_net(self):
        from akid.examples import alex_net
        kid = alex_net.setup()
        loss = kid.practice()
        assert loss < 3


if __name__ == "__main__":
    main()
