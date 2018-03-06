import unittest
import doctest
import os
import errno
import sys
sys.path.insert(0, '../')
import utils
from config import TEST_DIR


class TestSVD_SING(unittest.TestCase):
    def test_overall(self):
        def getTests(splits):
            splits[5] = round(float(splits[5]), 3)
            splits[8] = round(float(splits[8]), 3)
            tests = splits[:4] + splits[6:]
            return tests

        def clean(f, d):
            # Should be exactly one file under testdir
            os.remove(f)
            os.rmdir(d)

        try:
            # Try make test dir
            # Ref: https://stackoverflow.com/a/273227/9326078
            testdir = '../../' + TEST_DIR
            os.makedirs(testdir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e

        try:
            # Execute svd_sing.py by shell with TEST_SVD_SING on
            testFieldsNum = '10'
            testTrainNum = '10'
            testFoldNum = '2'
            testDataSet = 'ml-100k'
            rawFile = testdir + testFieldsNum + 'F' + testDataSet
            execStr = ' '.join(['DEBUG=WARNING TEST_SVD_SING=1',
                                'SVD_K_NUM=' + testFieldsNum,
                                'MAX_TRAIN_NUM=' + testTrainNum,
                                'LEARNING_RATE=0.001 LAMBDA=0.001 python2.7 svd_sing.py 0',
                                testFoldNum,
                                testDataSet])
            expectedExecStr = 'DEBUG=WARNING TEST_SVD_SING=1 SVD_K_NUM=10 MAX_TRAIN_NUM=10 LEARNING_RATE=0.001 LAMBDA=0.001 python2.7 svd_sing.py 0 2 ml-100k'
            self.assertEqual(execStr, expectedExecStr)
            os.system('cd ../ && ' + execStr)

            # Read rawFile (output) by 1st rows
            numRows = 12
            lines = []
            with open(rawFile, 'r') as f:
                for r in range(numRows):
                    lines.append(f.readline().strip())

            # Compare rawFile with expectations by prior rows
            expectedLines = [
                'ml-100k,10,0,0,4,inf,train,avgPrec,0.649345690819',
                'ml-100k,10,0,0,4,inf,train,microF1,0.581033262562',
                'ml-100k,10,0,0,4,inf,train,coverage,0.465990406542',
                'ml-100k,10,0,0,4,inf,train,hammingLoss,0.418966737438',
                'ml-100k,10,0,0,4,inf,train,RL,0.175188723756',
                'ml-100k,10,0,0,4,inf,train,oneError,0.0552016985138',
                'ml-100k,10,0,0,4,inf,valid,avgPrec,0.632048759239',
                'ml-100k,10,0,0,4,inf,valid,microF1,0.564265536723',
                'ml-100k,10,0,0,4,inf,valid,coverage,0.489092906466',
                'ml-100k,10,0,0,4,inf,valid,hammingLoss,0.435734463277',
                'ml-100k,10,0,0,4,inf,valid,RL,0.185587335217',
                'ml-100k,10,0,0,4,inf,valid,oneError,0.0677966101695',
            ]
            for ind, line in enumerate(lines):
                splitsExp = expectedLines[ind].split(',')
                splitsAct = line.split(',')
                testsExp = getTests(splitsExp)
                testsAct = getTests(splitsAct)
                self.assertEqual(testsExp, testsAct)
        except AssertionError:
            raise
        except OSError:
            raise

        # If we succeed
        clean(rawFile, testdir)


# Register for unittest's test discovery
def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(utils))
    return tests


if __name__ == '__main__':
    unittest.main()
