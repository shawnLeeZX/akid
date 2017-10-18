from akid import (
    Kid,
    FeedSensor,
    MomentumKongFu
)

from akid.utils.test import AKidTestCase, TestFactory, main
from akid import backend as A


class TestKongFu(AKidTestCase):
    def setUp(self):
        self.use_cuda_save = A.use_cuda()
        A.use_cuda(False)

        A.reset()

    def tearDown(self):
        A.use_cuda(self.use_cuda_save)

    def test_momentum_kongfu(self):
        from akid import initializers as init
        # import ipdb; ipdb.set_trace()
        a = A.get_variable(initializer=init.get('tensor', value=[1.]))
        b = 2
        c = a * b
        k = MomentumKongFu(var_list=[a], lr=1, name="test_kf")
        grad = k.forward(c)
        train_op = k.update(grad)

        A.init()
        if A.backend() == A.TF:
            A.run(train_op, feed_dict=k.get_feed_dict())
        a_eval = A.eval(a)

        self.assertEquals(-1, a_eval)

    # def test_placeholder_lr_scheme(self):
    #     from akid import LearningRateScheme
    #     brain = TestFactory.get_test_brain()
    #     source = TestFactory.get_test_feed_source()
    #     kid = Kid(
    #         FeedSensor(source_in=source, name='data'),
    #         brain,
    #         MomentumKongFu(lr_scheme={"name": LearningRateScheme.placeholder}),
    #         max_steps=900)

    #     def update_lr(kid):
    #         if kid.step < 200:
    #             kid.kongfu.lr_value = 0.1
    #         elif kid.step < 400:
    #             kid.kongfu.lr_value = 0.01
    #         elif kid.step < 600:
    #             kid.kongfu.lr_value = 0.001
    #         else:
    #             kid.kongfu.lr_value = 0.0001

    #     kid.hooks.on_batch_begin.append(update_lr)
    #     kid.setup()

    #     kid.practice()
    #     assert kid.kongfu.lr_value == 0.0001

    # def test_exp_decay_lr_scheme(self):
    #     from akid import LearningRateScheme
    #     brain = TestFactory.get_test_brain()
    #     source = TestFactory.get_test_feed_source()
    #     sensor = FeedSensor(source_in=source, name='data')
    #     kid = Kid(
    #         sensor,
    #         brain,
    #         MomentumKongFu(
    #             lr_scheme={
    #                 "name": LearningRateScheme.exp_decay,
    #                 "base_lr": 0.01,
    #                 "decay_rate": 0.95,
    #                 "num_batches_per_epoch": 468,
    #                 "decay_epoch_num": 1}),
    #         max_steps=900)

    #     kid.setup()
    #     kid.practice()

    #     with kid.sess as sess:
    #         lr = sess.run(kid.kongfu.learning_rate)
    #     assert abs(lr - 0.0095) <= 0.0001


if __name__ == "__main__":
    main()
