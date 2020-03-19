import unittest
import helper as h


class MyTestCase(unittest.TestCase):
    def test_add(self):
        self.assertEqual(h.add(1, 2), 3)


if __name__ == '__main__':
    unittest.main()
