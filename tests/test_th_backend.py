import os, unittest
os.environ['AKID_BACKEND'] = 'pytorch'
from akid.utils.test import TestSuite


if __name__ == '__main__':
    all_tests = TestSuite()
    unittest.main(defaultTest='all_tests.suite', exit=False)
