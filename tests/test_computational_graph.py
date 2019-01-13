import numpy as np

from akid.utils.test import AKidTestCase, main
from akid.layers import ConvolutionLayer
from akid import initializers
from akid import backend as A


class TestComputationalGraph(AKidTestCase):
    def setUp(self):
        A.reset()

    def test_var_scope(self):
        if A.DATA_FORMAT == "CHW":
            X_in = np.random.uniform(-1, 1, [1, 2, 3, 3])
        elif A.DATA_FORMAT == "HWC":
            X_in = np.random.uniform(-1, 1, [1, 3, 3, 2])
        else:
            raise ValueError("Data format {} not supported".format(A.DATA_FORMAT))
        l = ConvolutionLayer(ksize=[3, 3],
                             strides=[1, 1, 1, 1],
                             in_channel_num=2,
                             out_channel_num=1,
                             padding="VALID",
                             initial_bias_value=1.,
                             init_para={"name": "default"},
                             do_summary=False,
                             name="test_scope")
        l.forward(A.Tensor(X_in, requires_grad=True))

        assert A.is_name_the_same(A.get_name(l.var_list[0]), 'test_scope/weights'), \
            "{} is not the desired name {}".format(A.get_name(l.var_list[0]), 'test_scope/weights')
        assert A.is_name_the_same(A.get_name(l.var_list[1]), 'test_scope/biases'), \
            "{} is not the desired name {}".format(A.get_name(l.var_list[1]), 'test_scope/biases')

    def test_tensor_auto_name_cache(self):
        if A.DATA_FORMAT == "CHW":
            X_in = np.random.uniform(-1, 1, [1, 2, 3, 3])
        elif A.DATA_FORMAT == "HWC":
            X_in = np.random.uniform(-1, 1, [1, 3, 3, 2])
        else:
            raise ValueError("Data format {} not supported".format(A.DATA_FORMAT))
        l = ConvolutionLayer(ksize=[3, 3],
                             strides=[1, 1, 1, 1],
                             in_channel_num=2,
                             out_channel_num=1,
                             padding="VALID",
                             initial_bias_value=1.,
                             init_para={"name": "default"},
                             do_summary=False,
                             summarize_output=True,
                             name="test_cache")
        l.forward(A.Tensor(X_in, requires_grad=True))

        name_ref = 'test_cache/fmap' if A.backend() == A.TORCH else 'test_cache_1/fmap'
        assert A.is_name_the_same(A.get_name(l.data), name_ref), \
            "{} is not the desired name {}".format(A.get_name(l.data), name_ref)

    def test_save_restore_replaces_existing_variables(self):
        A.get_variable(name="x", initializer=np.array([0, 1]))
        A.init()
        A.save("x")
        A.restore("x")
        vars = A.get_all_variables()
        self.assertEquals(len(vars), 1)


if __name__ == "__main__":
    main()
