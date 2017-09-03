import sys, math
sys.path.insert(0, '../')
import unittest
import numpy as np
from run import getAvgPrecision, getCoverage

class TestRun(unittest.TestCase):
    # validate nothing 
    def test_nothing(self):
        self.assertEqual(2.0, 2.0)

    def test_getAvgPrecision(self):
        # given we have 3 + 3 + 2
        usr2NonzeroCols, usr2probs = {1: [0, 3, 6], 2: [0, 3, 6]}, {1: [0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.6, 0.4], 2: [0.3, 0.5, 0.2, 0.3, 0.2, 0.5, 0.4, 0.6]}
        avgPrec = getAvgPrecision(usr2NonzeroCols, usr2probs)
        expected = ((1/1 + 2/2 + 3/3)/ 3.0 + (1/4.0 + 2/5.0 + 3/6.0)/3.0 ) / 2.0 
        self.assertEqual(avgPrec, expected)

    def test_getCoverage(self):
        # given we have 3 + 3 + 2, total 8 fields
        totalFields = 8.0
        usr2NonzeroCols, usr2probs = {1: [0, 3, 6], 2: [0, 3, 6]}, {1: [0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.6, 0.4], 2: [0.3, 0.5, 0.2, 0.3, 0.2, 0.5, 0.4, 0.6]}
        cvgLoss = getCoverage(usr2NonzeroCols, usr2probs)
        expected = (3/totalFields + 6/totalFields)/ 2.0 
        self.assertEqual(cvgLoss, expected)

        

if __name__ == '__main__':
    unittest.main()
