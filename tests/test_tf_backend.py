from __future__ import absolute_import
import os, unittest
os.environ['AKID_BACKEND'] = 'tensorflow'
from akid.utils.test import TestSuite


if __name__ == '__main__':
    all_tests = TestSuite()
    unittest.main(defaultTest='all_tests.suite', exit=False)
