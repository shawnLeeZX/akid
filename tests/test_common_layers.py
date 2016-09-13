from akid.tests.test import AKidTestCase, main
import tensorflow as tf
from akid.layers import PaddingLayer


class TestCommonLayers(AKidTestCase):
    def test_padding(self):
        input = tf.constant(1, shape=[1, 2, 2, 1])
        layer = PaddingLayer(padding=[1, 1])
        layer.setup(input)
        assert layer.data.get_shape().as_list() == [1, 4, 4, 1]


if __name__ == "__main__":
    main()
