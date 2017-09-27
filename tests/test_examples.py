from akid.utils.test import AKidTestCase, main, debug_on


class TestExamples(AKidTestCase):
    def test_mnist_tf_tutorial(self):
        from akid.examples.mnist import mnist_tf_tutorial
        kid = mnist_tf_tutorial.setup()
        kid.max_steps = 4000
        loss, eval_ = kid.practice(return_eval=True)
        assert eval_[0] > 0.99, \
            "Critical failure. DO a full check."

    # def test_alex_net(self):
    #     from akid.examples import alex_net
    #     kid = alex_net.setup()
    #     kid.max_steps = 1000
    #     loss = kid.practice()
    #     assert loss < 3.2


if __name__ == "__main__":
    main()
