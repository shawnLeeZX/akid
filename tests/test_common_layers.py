from akid.utils.test import AKidTestCase, main
import tensorflow as tf
from akid.layers import PaddingLayer

from akid import backend as A


class TestCommonLayers(AKidTestCase):
    def setUp(self):
        A.reset()

    def test_padding(self):
        input = tf.constant(1, shape=[1, 2, 2, 1])
        layer = PaddingLayer(padding=[1, 1])
        layer.forward(input)
        assert layer.data.get_shape().as_list() == [1, 4, 4, 1]

    def test_scattering(self):
        input = tf.constant(1, shape=[2, 5])
        from akid.layers import ScatterLayer

        scatter_len_list = [1, 2]
        layer = ScatterLayer(scatter_len_list=scatter_len_list)
        layer.forward(input)
        scattered_list = layer.data
        for i, t in enumerate(scattered_list):
            shape = t.get_shape().as_list()
            assert shape[-1] == scatter_len_list[i]


if __name__ == "__main__":
    main()
