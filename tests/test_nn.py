import numpy as np

from akid.utils.test import AKidTestCase, main, skipUnless
from akid import backend as A

class TestOps(AKidTestCase):
    @skipUnless(A.backend() == A.TORCH, "Some functions used are not implemented in tensorflow yet.")
    def test_hessian(self):
        x = A.Tensor(A.ones(1), requires_grad=True)
        y = A.Tensor(A.ones(1), requires_grad=True)
        z = x ** 2 * y ** 2
        H = A.nn.hessian(z, [x, y])
        H_ref = np.array([[2, 4],
                          [4, 2]])
        self.assertNdarrayEquals(A.eval(H), H_ref)


if __name__ == "__main__":
    main()
