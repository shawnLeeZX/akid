import numpy as np

from akid.utils.test import AKidTestCase, main
from akid.layers import ConvolutionLayer
from akid import backend as A


class TestComputationalGraph(AKidTestCase):
    def test_var_scope(self):
        X_in = np.random.uniform(-1, 1, [1, 2, 3, 3])
        l = ConvolutionLayer(ksize=[3, 3],
                             strides=[1, 1, 1, 1],
                             in_channel_num=2,
                             out_channel_num=1,
                             padding="VALID",
                             initial_bias_value=1.,
                             init_para={"name": "default"},
                             do_summary=False,
                             name="test_scope")
        l.forward(A.Tensor(X_in, require_grad=True))

        self.assertEqual(A.get_name(l.var_list[0]), 'test_scope/weights')
        self.assertEqual(A.get_name(l.var_list[1]), 'test_scope/biases')

    def test_tensor_auto_name_cache(self):
        X_in = np.random.uniform(-1, 1, [1, 2, 3, 3])
        l = ConvolutionLayer(ksize=[3, 3],
                             strides=[1, 1, 1, 1],
                             in_channel_num=2,
                             out_channel_num=1,
                             padding="VALID",
                             initial_bias_value=1.,
                             init_para={"name": "default"},
                             do_summary=False,
                             name="test_cache")
        l.forward(A.Tensor(X_in, require_grad=True))

        self.assertEqual(A.get_name(l.data), 'test_cache/fmapd')


if __name__ == "__main__":
    main()
