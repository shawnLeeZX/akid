from akid.utils.test import AKidTestCase, main, TestFactory, skipUnless
from akid import backend as A

class TestEvalLayers(AKidTestCase):
    def setUp(self):
        self.use_cuda = A.use_cuda()
        A.use_cuda(False)

    def tearDown(self):
        A.use_cuda(self.use_cuda)

    def test_multilabel_accuracy_layer(self):
        y = A.Tensor([[1, 1], [0, 1]])
        y_pred = A.Tensor([[0.1, 0.9], [0.1, 0.1]])

        from akid.layers import MultiLabelAccuracy
        l = MultiLabelAccuracy()
        acc = l.forward([y_pred, y])
        acc = A.eval(acc)
        self.assertEquals(acc, 0.75)


if __name__ == "__main__":
    main()
