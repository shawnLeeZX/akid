from akid import (
    Kid,
    FeedSensor,
    MomentumKongFu,
)

from akid.utils.test import AKidTestCase, TestFactory, main


class TestLog(AKidTestCase):
    """
    NOTE: Ideally, the test should reset the logging stream and examine the
    output pattern, as in the [logging
    test](http://svn.python.org/view/python/trunk/Lib/test/test_logging.py?view=markup)
    in the official logging python library. Now I just manually check the
    results ...
    """
    def test_logging(self):
        brain = TestFactory.get_test_brain()
        source = TestFactory.get_test_feed_source()
        kid = Kid(
            FeedSensor(source_in=source, name='data'),
            brain,
            MomentumKongFu(name="opt"),
            engine="data_parallel",
            max_steps=1000)
        kid.setup()

        loss = kid.practice()
        assert loss < 3

if __name__ == "__main__":
    main()
