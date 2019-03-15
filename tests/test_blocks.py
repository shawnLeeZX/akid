from akid.utils.test import AKidTestCase, main


class TestBlocks(AKidTestCase):
    def test_auto_naming(self):
        from akid.layers import MSELossLayer

        l = MSELossLayer()
        self.assertEquals("{}_{}".format(MSELossLayer.NAME, 1), l.name)


if __name__ == "__main__":
    main()
