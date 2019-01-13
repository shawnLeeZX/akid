import numpy as np

from akid.utils.test import AKidTestCase, main, debug_on, skipUnless
from akid.layers import ReLULayer, MaxPoolingLayer
from akid import backend as A


from akid.utils import glog as log
log.init()


class TestSystem(AKidTestCase):
    def setUp(self):
        self.use_cuda_save = A.use_cuda()
        A.use_cuda(False)

    def tearDown(self):
        A.use_cuda(self.use_cuda_save)
        A.reset()

    def test_sequential_system(self):
        from akid import SequentialSystem
        s = SequentialSystem(name="test_sequential_system")
        s.attach(MaxPoolingLayer(ksize=[2, 2],
                                 strides=[1, 1, 1, 1],
                                 padding="VALID",
                                 name="max_pool"))
        s.attach(ReLULayer(name="relu"))


        X_in = np.array([[[1., 2],
                          [3, 4]],
                         [[-5, -6],
                          [-7, -8]]], dtype=np.float32)
        X_in = np.expand_dims(X_in, axis=0)
        X_in = A.standardize_data_format(X_in, old_format='nchw')

        X_f_out_ref = np.array([[[4.]], [[0]]], dtype=np.float32)
        X_f_out_ref = np.expand_dims(X_f_out_ref, axis=0)
        X_f_out_ref = A.standardize_data_format(X_f_out_ref, old_format='nchw')

        X_f_out = s.forward(A.Tensor(X_in))

        A.init()
        X_f_out_eval = A.eval(X_f_out)

        assert (X_f_out_eval == X_f_out_ref).all(), \
            "X_f_out_eval: {}; X_f_out_ref{}".format(X_f_out_eval, X_f_out_ref)

    @skipUnless(A.backend() == A.TF)
    def test_sequential_g_system(self):
        from akid import SequentialGSystem
        s = SequentialGSystem(name="test_sequential_g_system")
        s.attach(MaxPoolingLayer(ksize=[2, 2],
                                 strides=[1, 1, 1, 1],
                                 padding="VALID",
                                 get_argmax_idx=True,
                                 name="max_pool"))
        s.attach(ReLULayer(name="relu"))


        X_in = np.array([[[1., 2],
                          [3, 4]],
                         [[-5, -6],
                          [-7, -8]]], dtype=np.float32)
        X_in = np.expand_dims(X_in, axis=0)
        X_in = A.standardize_data_format(X_in, old_format='nchw')

        X_f_out_ref = np.array([[[4.]], [[0]]], dtype=np.float32)
        X_f_out_ref = np.expand_dims(X_f_out_ref, axis=0)
        X_f_out_ref = A.standardize_data_format(X_f_out_ref, old_format='nchw')

        X_b_out_ref = np.array([[[0., 0],
                                 [0, 4]],
                                [[0, 0],
                                 [0, 0]]], dtype=np.float32)
        X_b_out_ref = np.expand_dims(X_b_out_ref, axis=0)
        X_b_out_ref = A.standardize_data_format(X_b_out_ref, old_format='nchw')

        X_f_out = s.forward(A.Tensor(X_in))
        X_b_out = s.backward(X_f_out)

        A.init()
        X_f_out_eval = A.eval(X_f_out)
        X_b_out_eval = A.eval(X_b_out)

        assert (X_f_out_eval == X_f_out_ref).all(), \
            "X_f_out_eval: {}; X_f_out_ref{}".format(X_f_out_eval, X_f_out_ref)
        assert (X_b_out_eval == X_b_out_ref).all(), \
            "X_b_out_eval: {}; X_b_out_ref{}".format(X_b_out_eval, X_b_out_ref)

    def test_graph_system(self):
        X_in = A.Tensor([[1, 0], [0, 0]], requires_grad=True)
        label_in = A.Tensor([[1, 0], [0, 0]], requires_grad=True)

        weight = A.Tensor([
            [1, 1],
            [0, 0]
        ])
        label_in = label_in + 1

        from akid import GraphSystem
        from akid.layers import InnerProductLayer, MSELossLayer
        s = GraphSystem(name="test_graph_system")
        s.attach(
            InnerProductLayer(in_channel_num=2,
                              out_channel_num=2,
                              initial_bias_value=1.,
                              init_para={"name": "tensor",
                                         "value": weight},
                              name='ip'))
        s.attach(MSELossLayer(name='loss'))
        X_out = s.forward([X_in, label_in])
        X_out_ref = 1

        A.init()
        X_out_eval = A.eval(X_out[0])
        self.assertEquals(X_out_eval, X_out_ref)

if __name__ == "__main__":
    main()
