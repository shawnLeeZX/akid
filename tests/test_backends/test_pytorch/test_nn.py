from __future__ import absolute_import
import numpy as np

from akid.utils.test import AKidTestCase, TestFactory, main, debug_on
from akid import backend as A


class TestNn(AKidTestCase):
    def test_nn_riemannic_metric_4d(self):
        # First test K = None
        W = A.Tensor([
            [
                [
                    [1, 0],
                    [0, 1]
                ],
                [
                    [-1, 0],
                    [0, -1]
                ]
            ],
            [
                [
                    [1, 0],
                    [0, 1]
                ],
                [
                    [-1, 0],
                    [0, -1]
                ]
            ]
        ])
        b = A.Tensor([1, -1])
        K = None
        out_ref = np.array([[5, 3],
                            [3, 5]])

        out = A.nn.nn_riemannic_metric(K, W, b)

        out_eval = A.eval(out)

        # Test the case with no bias

        out_ref = np.array([[4, 4],
                            [4, 4]])

        out = A.nn.nn_riemannic_metric(K, W, None)

        out_eval = A.eval(out)

        self.assertNdarrayEquals(out_eval, out_ref)

        # Test with a metric K
        K = A.Tensor([[1, -1],
                      [-1, 1]])
        out_ref = np.array([[9, 7],
                            [7, 9]])

        out = A.nn.nn_riemannic_metric(K, W, b)

        out_eval = A.eval(out)

        self.assertNdarrayEquals(out_eval, out_ref)

    def test_nn_riemannic_metric_2d(self):
        # First test K = None
        W = A.Tensor([[-1, 1],
                      [-1, 1]])
        b = A.Tensor([1, -1])
        K = A.Tensor([[1, -1],
                      [-1, 1]])
        out_ref = np.array([[1, 0],
                            [0, 1]])

        out = A.nn.nn_riemannic_metric(K, W, b)

        out_eval = A.eval(out)

        self.assertNdarrayEquals(out_eval, out_ref)

        # Test the case with no bias

        out_ref = np.array([[0, 0],
                            [0, 0]])
        out = A.nn.nn_riemannic_metric(K, W, None)
        out_eval = A.eval(out)
        self.assertNdarrayEquals(out_eval, out_ref)

    def test_normalize_weight_4d(self):
        W = A.Tensor(np.ones((2, 2, 2, 2)))
        W = A.nn.normalize_weight(W)

        W_ref = A.Tensor(np.ones((2, 2, 2, 2)) * np.sqrt(0.125))

        self.assertTensorEquals(W, W_ref)

    def test_normalize_weight_2d(self):
        W = A.Tensor(np.ones((2, 2)))
        W = A.nn.normalize_weight(W)

        W_ref = A.Tensor(np.ones((2, 2)) * np.sqrt(1./2))

        self.assertTensorEquals(W, W_ref)

if __name__ == "__main__":
    main()
