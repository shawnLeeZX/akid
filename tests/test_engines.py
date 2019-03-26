from __future__ import absolute_import
from akid import (
    Kid,
    FeedSensor,
    MomentumKongFu,
    MNISTFeedSource,
    AKID_DATA_PATH
)
from akid.models import OneLayerBrain

from akid.utils.test import AKidTestCase, TestFactory, main, skipUnless, debug_on
from akid import backend as A
from six.moves import range


class TestEngine(AKidTestCase):
    def setUp(self):
        A.reset()

    @skipUnless(A.backend() == A.TORCH)
    def test_data_parallel_simple(self):
        from akid import GraphBrain
        from akid.layers import InnerProductLayer, MSELossLayer
        from akid import engines

        brain = GraphBrain(name="brain")
        weight = A.Tensor([2])
        l = InnerProductLayer(in_channel_num=1,
                              out_channel_num=1,
                              init_para={"name": "tensor",
                                         "value": weight},
                              initial_bias_value=None,
                              name="ip")
        brain.attach(l)
        brain.attach(MSELossLayer(inputs=[{"name": "ip"},
                                          {"name": "system_in", 'idxs': [1]}],
                                  name="loss"))
        kongfu = MomentumKongFu(name="opt")

        engine = engines.get(brain=brain,
                             kongfu=kongfu,
                             # name="single",
                             name="data_parallel",
                             gpu_num=2
        )
        engine.setup()

        data = A.Tensor([[1], [2]], requires_grad=True).cuda()
        labels = A.Tensor([[3], [3]], requires_grad=True).cuda()
        loss = engine.forward([data, labels])

        A.init()
        self.assertEqual(A.eval(loss), 1)

        grad = kongfu.forward(loss)
        if A.backend() == A.TORCH:
            grad = l.weights.grad

        self.assertEqual(1, A.eval(grad))

    @skipUnless(A.backend() == A.TORCH)
    def test_data_parallel_complex(self):
        # Test with real world training.
        brain = OneLayerBrain(name="brain")
        for b in brain.blocks:
            b.wd = {"type": "l2", "scale": 5e-4}

        sensor = TestFactory.get_test_sensor()
        kid = Kid(
            sensor,
            brain,
            MomentumKongFu(name="opt"),
            engine={"name": "data_parallel", "gpu_num": 2},
            max_steps=1000)
        kid.setup()

        brain = OneLayerBrain(name="brain")
        for b in brain.blocks:
            b.wd = {"type": "l2", "scale": 5e-4}
        sensor = TestFactory.get_test_sensor()
        kid_ref = Kid(
            sensor,
            brain,
            MomentumKongFu(name="opt"),
            engine={"name": "single"},
            max_steps=1000)
        kid_ref.setup()

        self.assertTensorAlmostEquals(kid.brain.blocks[0].biases,
                                      kid_ref.brain.blocks[0].biases)
        self.assertTensorEquals(kid.sensor.training_data,
                                kid_ref.sensor.training_data)

        for i in range(10):
            kid.step()
            kid_ref.step()

            self.assertTensorEquals(kid.engine.towers[1].blocks[0].biases,
                                    kid.engine.towers[0].blocks[0].biases)

        self.assertTensorAlmostEquals(kid.brain.blocks[0].biases,
                                      kid_ref.brain.blocks[0].biases)

    @skipUnless(A.backend() == A.TORCH)
    def test_data_parallel_train(self):
        # Test with real world training.
        brain = OneLayerBrain(name="brain")
        for b in brain.blocks:
            b.summarize_output = True
            b.wd = None

        sensor = TestFactory.get_test_sensor()
        kid = Kid(
            sensor,
            brain,
            MomentumKongFu(name="opt"),
            engine={"name": "data_parallel", "gpu_num": 2},
            max_steps=1000)

        kid.setup()

        loss = kid.practice()
        self.assertLess(loss, 0.2)



if __name__ == "__main__":
    main()
