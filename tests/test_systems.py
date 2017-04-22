import numpy as np

from akid.utils.test import AKidTestCase, main
from akid.layers import ReLULayer, MaxPoolingLayer
from akid import backend as A


from akid.utils import glog as log
log.init()


class TestSystem(AKidTestCase):
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
        X_in = np.einsum("chw->hwc", X_in)
        X_in = np.expand_dims(X_in, axis=0)

        X_f_out_ref = np.array([[[4.]], [[0]]], dtype=np.float32)
        X_f_out_ref = np.einsum("chw->hwc", X_f_out_ref)
        X_f_out_ref = np.expand_dims(X_f_out_ref, axis=1)

        X_b_out_ref = np.array([[[0., 0],
                                 [0, 4]],
                                [[0, 0],
                                 [0, 0]]], dtype=np.float32)
        X_b_out_ref = np.einsum("chw->hwc", X_b_out_ref)
        X_b_out_ref = np.expand_dims(X_b_out_ref, axis=0)

        X_f_out = s.forward(A.Tensor(X_in))
        X_b_out = s.backward(X_f_out)

        A.init()
        X_f_out_eval = A.eval(X_f_out)
        X_b_out_eval = A.eval(X_b_out)

        assert (X_f_out_eval == X_f_out_ref).all(), \
            "X_f_out_eval: {}; X_f_out_ref{}".format(X_f_out_eval, X_f_out_ref)
        assert (X_b_out_eval == X_b_out_ref).all(), \
            "X_b_out_eval: {}; X_b_out_ref{}".format(X_b_out_eval, X_b_out_ref)


if __name__ == "__main__":
    main()
