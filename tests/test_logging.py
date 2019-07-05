from __future__ import absolute_import
from akid import (
    Kid,
    FeedSensor,
    MomentumKongFu,
)

from akid.utils.test import AKidTestCase, TestFactory, main, skipUnless

from akid import backend as A


class TestLog(AKidTestCase):
    """
    NOTE: Ideally, the test should reset the logging stream and examine the
    output pattern, as in the [logging
    test](http://svn.python.org/view/python/trunk/Lib/test/test_logging.py?view=markup)
    in the official logging python library. Now I just manually check the
    results ...
    """
    @skipUnless(A.backend() == A.TORCH)
    def test_logging(self):
        brain = TestFactory.get_test_brain()
        sensor = TestFactory.get_test_sensor()
        kid = Kid(
            sensor,
            brain,
            MomentumKongFu(name="opt"),
            engine={"name": "data_parallel"},
            max_steps=1000)
        kid.setup()

        loss = kid.practice()
        assert loss < 3

        kid.teardown()

if __name__ == "__main__":
    main()
