from __future__ import print_function

from __future__ import absolute_import
import tensorflow as tf
import numpy as np

from akid.utils.test import AKidTestCase, main, skipUnless
from akid.ops.random_ops import msra_initializer
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


if __name__ == "__main__":
    main()
