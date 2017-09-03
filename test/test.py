import sys, math
sys.path.insert(0, '../')
import unittest
import numpy as np
from run import getAvgPrecision, getCoverage, getMicroF1ByCol, getOneError, getRL
from config import USR_TOTAL_LABELS_FIELDS

class TestRun(unittest.TestCase):

    # validate getAvgPrecision
    def test_getAvgPrecision(self):
        # given we have 3 + 3 + 2
        usr2NonzeroCols, usr2probs = {1: [0, 3, 6], 2: [0, 3, 6]}, {1: [0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.6, 0.4], 2: [0.3, 0.5, 0.2, 0.3, 0.2, 0.5, 0.4, 0.6]}
        avgPrec = getAvgPrecision(usr2NonzeroCols, usr2probs)
        expected = ((1/1 + 2/2 + 3/3)/ 3.0 + (1/4.0 + 2/5.0 + 3/6.0)/3.0 ) / 2.0 
        self.assertEqual(avgPrec, expected)

    # validate getCoverage
    def test_getCoverage(self):
        # given we have 3 + 3 + 2, total 8 fields
        totalFields = 8.0
        usr2NonzeroCols, usr2probs = {1: [0, 3, 6], 2: [0, 3, 6]}, {1: [0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.6, 0.4], 2: [0.3, 0.5, 0.2, 0.3, 0.2, 0.5, 0.4, 0.6]}
        cvgLoss = getCoverage(usr2NonzeroCols, usr2probs)
        expected = (3/totalFields + 6/totalFields)/ 2.0 
        self.assertEqual(cvgLoss, expected)

    # validate getMicroF1ByCol
    def test_getMicroF1ByCol(self):
	# say, it's 2 + 2 + 5 attrs
        u2predictions = {
	  0: [0, 2, 4],
	  1: [1, 3, 5],
	  2: [1, 2, 6],
	}

	usr2NonzeroCols = {
          0: [0, 3, 5],
          1: [1, 2, 8],
          2: [0, 3, 6],
        }

        expectMicroF1 = 1.0 / 3
        actualMicroF1 = getMicroF1ByCol(u2predictions, usr2NonzeroCols)
        self.assertEqual(expectMicroF1, actualMicroF1)

    # validate getOneError
    def test_getOneError(self):
	# say, it's 2 + 2 + 5 attrs
        u2predictions = {
	  0: [0, 2, 4],
	  1: [1, 3, 5],
	  2: [1, 2, 6],
	}

	usr2NonzeroCols = {
          0: [0, 3, 5],
          1: [1, 2, 8],
          2: [0, 3, 7],
        }

        expectOneError = 1/3.0
        actualOneError = getOneError(u2predictions, usr2NonzeroCols)
        self.assertEqual(expectOneError, actualOneError)

    # validate getRL
    def test_getRL(self):
        # say, it's 2 + 2 + 5 attrs
        u2predictions = {
	  0: [0, 2, 4],
	  1: [1, 3, 5],
	}

	usr2NonzeroCols = {
          0: [0, 3, 5],
          1: [1, 2, 8],
        }
        
	combNums = float( (2+2+5 - 3) * 3 )
        expectRL = ((2+2+2+0+0+0)/combNums + (2+2+1+2+1+1)/combNums) / 2
        actualRL = getRL(u2predictions, usr2NonzeroCols)
        self.assertTrue( expectRL * combNums - actualRL *USR_TOTAL_LABELS_FIELDS * (USR_TOTAL_LABELS_FIELDS - len(usr2NonzeroCols.itervalues().next())) < 0.001)

if __name__ == '__main__':
    unittest.main()
