from akid.tests.test import TestCase, TestFactory, main


class TestBrain(TestCase):
    def test_moving_average(self):
        brain = TestFactory.get_test_brain(using_moving_average=True)
        source = TestFactory.get_test_feed_source()
        kid = TestFactory.get_test_survivor(source, brain)
        kid.setup()

        precision = kid.practice()
        assert precision >= 0.95

    def assert_diff(self, brain_a, brain_b):
        """
        Compare two brains, and make sure they are completely different two.
        """
        for block_a, block_b in zip(brain_a.blocks, brain_b.blocks):
            assert block_a != block_b
        # Now loss layer is treated specially, so we need to test it specially.
        assert brain_a.loss_layer != brain_b.loss_layer

    def test_copy(self):
        brain = TestFactory.get_test_brain(using_moving_average=True)
        brain_copy = brain.get_copy()
        self.assert_diff(brain, brain_copy)

    def test_val_copy(self):
        brain = TestFactory.get_test_brain(using_moving_average=True)
        val_brain = brain.get_val_copy()
        self.assert_diff(brain, val_brain)
        for b in val_brain.blocks:
            assert b.is_val is True
        assert val_brain.loss_layer.is_val is True

if __name__ == "__main__":
    main()
