from __future__ import print_function

from __future__ import absolute_import
import tensorflow as tf
import numpy as np

from akid.utils.test import AKidTestCase, main, skipUnless, TestFactory, debug_on
from akid.ops.random_ops import msra_initializer
from akid.ops import matrix_ops
from akid import backend as A


class TestOps(AKidTestCase):
    def setUp(self):
        super(TestOps, self).setUp()
        self.graph = tf.Graph()

    @skipUnless(A.backend() == A.TF)
    def test_msra_init_op_conv(self):
        with self.graph.as_default():
            var = tf.get_variable("init_test_variable",
                                  [5, 5, 12, 24],
                                  initializer=msra_initializer(factor=1))
            mean = tf.reduce_mean(var, [0, 1, 3])
        with tf.Session(graph=self.graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            mean = mean.eval()
            var = var.eval()
            var = np.einsum("hwcn->hwnc", var)
            variance_scaling = np.mean(np.sum((var - mean)**2, (0, 1, 2)))
            print("gradient variance scaling factor for the weight is {}"
                  ".".format(variance_scaling))
            assert variance_scaling > 1 and variance_scaling < 2

    @skipUnless(A.backend() == A.TF)
    def test_msra_init_op_ip(self):
        with self.graph.as_default():
            var = tf.get_variable("init_test_variable", [100, 1000],
                                  initializer=msra_initializer(factor=1))
            mean = tf.reduce_mean(var, [1])
        with tf.Session(graph=self.graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            mean = mean.eval()
            var = var.eval()
            var = np.einsum("cn->nc", var)
            variance_scaling = np.mean(np.sum((var - mean)**2, (0)))
            print("gradient variance scaling factor for the weight is {}"
                  ".".format(variance_scaling))
            assert variance_scaling > 1 and variance_scaling < 2


    @skipUnless(A.backend() == A.TORCH, "Some functions used are not implemented in tensorflow yet.")
    def test_lanczos_tridiagonalization_slow(self):
        import torch as th
        H = A.randn((1000, 1000))
        H = H @ H.t()
        H = A.cast(H, th.float64) # TODO: fix here
        eig = matrix_ops.lanczos_tridiagonalization_slow(H)[0]
        eig_ref = A.symeig(H)[0]
        self.assertNdarrayAlmostEquals(A.eval(eig), A.eval(eig_ref), places=5)


    @skipUnless(A.backend() == A.TORCH, "Some functions used are not implemented in tensorflow yet.")
    def test_lanczos_tridiagonalization_fast(self):
        H = A.randn((5, 5))
        H = H @ H.t()
        eig = matrix_ops.lanczos_tridiagonalization_fast(H)[0]
        eig_ref = A.symeig(H)[0]
        self.assertNdarrayAlmostEquals(A.eval(eig), A.eval(eig_ref), places=5)


    @skipUnless(A.backend() == A.TORCH, "Some functions used are not implemented in tensorflow yet.")
    def test_lanczos_spectrum_approx(self):
        N = 1000
        H = np.random.randn(N, N)
        H = H + H.T - np.diag(H.diagonal())
        H = A.Tensor(H)
        # A PSD matrix can be created as follows, though is not used in the test.
        # H = H @ H.t()
        eigenvalues = A.symeig(H)[0]
        spectrum_norm = A.max(eigenvalues)
        H /= spectrum_norm

        K = 1024
        n_vec = 1
        eigs = matrix_ops.lanczos_spectrum_approx(H, 100, K, n_vec)
        eig_ref = A.symeig(H)[0]
        import seaborn as sns
        from matplotlib import pyplot as plt
        import pandas as pd
        plt.figure()
        sns.distplot(A.eval(eig_ref), bins=50, norm_hist=True, kde=False)
        sns.lineplot(data=pd.DataFrame(A.eval(eigs), index=np.linspace(-1, 1, K)) )
        plt.savefig("lanczos_wigner.jpg")

    @skipUnless(A.backend() == A.TORCH, "Some functions used are not implemented in tensorflow yet.")
    def test_lanczos_spectrum_approx_nn(self):
        brain = TestFactory.get_test_brain()
        sensor = TestFactory.get_test_sensor()
        sensor.set_batch_size(512)
        sensor.num_workers = 8
        kid = TestFactory.get_test_kid(sensor, brain)
        kid.do_summary = False
        kid.setup()
        kid.sensor.set_mode("train")
        kid.sensor.setup()
        from akid.core.callbacks import on_check
        psi, c, d = on_check(kid)
        psi, c, d = A.eval(psi), A.eval(c), A.eval(d)

        import seaborn as sns
        from matplotlib import pyplot as plt
        import pandas as pd
        plt.figure()
        sns.semilogy(data=pd.DataFrame(psi, index=np.linspace(-d + c, d, 400)) )
        plt.savefig("lanczos_nn.jpg")

    def test_center_unit_eig_normalization(self):
        H = np.random.randn(1000, 1000)
        H = H + H.T - np.diag(H.diagonal())
        H = A.Tensor(H)

        H, _, _ = matrix_ops.center_unit_eig_normalization(H, 100, 0.05)

        eig_ref = A.symeig(H)[0]
        self.assertLess(eig_ref[-1], 1)
        self.assertGreater(eig_ref[-1], 0.8)
        self.assertLess(eig_ref[0], -0.8)
        self.assertGreater(eig_ref[0], -1)

if __name__ == "__main__":
    main()
