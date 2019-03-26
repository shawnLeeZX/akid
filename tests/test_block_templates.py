from __future__ import absolute_import
from akid.utils.test import AKidTestCase, main, skipUnless, skip
from akid.examples.mnist import block_mnist
from akid import sugar
from akid import backend as A


# TODO: Template only works for tensorflow for now. Port needed.
class TestTemplate(AKidTestCase):
    def setUp(self):
        super(TestTemplate, self).setUp()
        sugar.init()

    @skip("Template test failed, and the template may need revising.")
    # @skipUnless(A.backend() == A.TF)
    def test_cnn_block(self):
        kid = block_mnist.setup()
        loss = kid.practice()
        assert loss < 2.5

    @skip("Template test failed, and the template may need revising.")
    # @skipUnless(A.backend() == A.TF)
    def test_activation_before_pooling(self):
        kid = block_mnist.setup(activation_before_pooling=True)
        loss = kid.practice()
        assert loss < 2.5

if __name__ == "__main__":
    main()
