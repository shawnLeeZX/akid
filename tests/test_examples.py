from __future__ import absolute_import
from akid.utils.test import AKidTestCase, main, debug_on, skipUnless

from akid import backend as A


class TestExamples(AKidTestCase):
    def setUp(self):
        A.reset()

    def test_mnist_tf_tutorial(self):
        from akid.examples.mnist import mnist_tf_tutorial
        kid = mnist_tf_tutorial.setup()
        kid.max_steps = 4000
        loss, eval_ = kid.practice(return_eval=True)
        assert eval_[0] > 0.99, \
            "Critical failure. DO a full check."

    @skipUnless(A.backend() == A.TF, "The data augmentation code has not been ported to CIFAR10")
    def test_alex_net(self):
        from akid.examples import alex_net
        kid = alex_net.setup()
        kid.max_steps = 1000
        loss = kid.practice()
        assert loss < 3.2


if __name__ == "__main__":
    main()
