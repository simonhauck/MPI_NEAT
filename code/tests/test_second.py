import unittest
import helper as h
from test_class import TestClass


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(h.subtract(3, 2), 1)

    def test_testclass(self):
        testclass = TestClass()
        self.assertEqual(testclass.add_class(3, 5), 8)


if __name__ == '__main__':
    unittest.main()
