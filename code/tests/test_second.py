import unittest
import helper as h


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(h.subtract(3, 2), 1)


if __name__ == '__main__':
    unittest.main()
