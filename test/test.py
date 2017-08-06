import sys, math
sys.path.insert(0, '../')
import unittest
import numpy as np

class TestRun(unittest.TestCase):
    # validate nothing 
    def test_nothing(self):
        self.assertEqual(2.0, 2.0)

if __name__ == '__main__':
    unittest.main()
