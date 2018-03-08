from scipy.sparse import csc_matrix  # For containing parse matrix
from itertools import product


def splitKfolds(usrsNum, k, shuffle):
    ''' Split to k ndarrays of usrs' indice
    >>> import random
    >>> random.seed(123)
    >>> usrNum, foldNum = 10, 2
    >>> kf = splitKfolds(usrNum, foldNum, random.shuffle)
    >>> print kf
    [[7, 1, 4, 2, 6], [5, 8, 3, 9, 0]]
    '''
    def partition(lst, n):
        '''Simple partition code
        ref: https://stackoverflow.com/questions/3352737/python-randomly-partition-a-list-into-n-nearly-equal-parts
        '''
        division = len(lst) / float(n)
        return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n)]

    usrsIndice = range(usrsNum)
    shuffle(usrsIndice)  # Should be in-place shuffle
    Kfolds = partition(usrsIndice, k)
    return Kfolds


def getTrainValid(kfolds, foldNum, usrs, items, ratings):
    '''get train/valid's corresponding data
    usrs should have unique usr numbers
    >>> import numpy as np
    >>> kfolds = [[0, 2], [1, 3], [5, 4]]
    >>> foldNum = 1
    >>> usrs = np.array([1, 2, 3, 4, 5, 6])
    >>> items = np.array([6, 5, 4, 3, 2, 1])
    >>> ratings = np.array([3.0, 1.0, 3.0, 3.0, 2.0, 3.0])
    >>> print str(getTrainValid(kfolds, foldNum, usrs, items, ratings)).replace(' ', '')
    (array([3.,3.,3.,2.]),array([1.,3.]),[1,3,6,5],[2,4],array([6,4,1,2]),array([5,3]))
    '''
    validIndices = kfolds[foldNum]
    trainIndices = [item for sublist in kfolds[:foldNum] + kfolds[foldNum + 1:] for item in sublist]

    return ratings[trainIndices], ratings[validIndices], \
        usrs[trainIndices].tolist(), usrs[validIndices].tolist(), \
        items[trainIndices], items[validIndices]


def loadToCSCMatrix(ratings, ratingUsrs, uniqShuffledUsrs, items):
    '''Load ratings to scipy's sparse matrix with unrevealed entries filled with zeros
    loaded matrix should be len(uniqUsrs) X len(items) and sparse
    >>> import numpy as np
    >>> ratings = np.array([1.0, 2.0, 2.0, 4.0])
    >>> ratingUsrs = np.array([3, 14, 3, 14])
    >>> uniqShuffledUsrs = np.unique(ratingUsrs).tolist()
    >>> items = np.array([8, 4, 3, 8])
    >>> print str(loadToCSCMatrix(ratings, ratingUsrs, uniqShuffledUsrs, items).toarray()).replace(' ', '')
    [[0.0.0.2.0.0.0.0.1.]
    [0.0.0.0.2.0.0.0.4.]]
    '''
    # Index each rating's usr by uniqShuffledUsrs
    # uniqShuffledUsrs: ordered uniq usrs in train's and valid's
    ratingsUsrs = [uniqShuffledUsrs.index(usr) for usr in ratingUsrs]
    return csc_matrix((ratings, (ratingsUsrs, items)))


def getDataStats(usrs4Train, usrs4Valid):
    '''Get data's stats container
    >>> usrs4Train, usrs4Valid = [0, 2, 4], [1, 3]
    >>> dataStats = getDataStats(usrs4Train, usrs4Valid)
    >>> print dataStats[0]
    {'u2probs': {0: [], 2: [], 4: []}, 'u2predictions': {0: [], 2: [], 4: []}, 'name': 'train', 'usrs': [0, 2, 4]}
    >>> print dataStats[1]
    {'u2probs': {1: [], 3: []}, 'u2predictions': {1: [], 3: []}, 'name': 'valid', 'usrs': [1, 3]}
    '''
    def getTemplate(dataset):
        usrs = usrs4Train if dataset == 'train' else usrs4Valid
        u2predictions = dict((usr, []) for usr in usrs)
        u2probs = dict((usr, []) for usr in usrs)
        return {
            'name': dataset,
            'usrs': usrs,
            'u2probs': u2probs,
            'u2predictions': u2predictions,
        }

    return [getTemplate(dataset) for dataset in ['train', 'valid']]


def handlePrediction(dataStats, uTrain, uValid, logreg):
    '''handle 'per-attribute' prediction tasks
    '''
    for d in dataStats:
        # Get corresponding uMatrix
        dataset = d['name']
        uMatrix = uTrain if dataset == 'train' else uValid

        # Predict
        predictions = logreg.predict(uMatrix)
        probs = logreg.predict_proba(uMatrix)

        # Add to result
        usrsCnt = uMatrix.shape[0]
        for ind in range(usrsCnt):
            usr = d['usrs'][ind]
            d['u2predictions'][usr].append(predictions[ind])
            d['u2probs'][usr] += probs[ind].tolist()


def getSortedPos(vals, reverse=False):
    '''Get sorted positions
    >>> vals = [-1, 8, 1, 0.3]
    >>> print str(getSortedPos(vals, reverse=True)).replace(' ', '')
    [1,2,3,0]
    '''
    posVals = [(ind, prob) for ind, prob in enumerate(vals)]
    posVals.sort(key=lambda tup: tup[1], reverse=reverse)
    return [tup[0] for tup in posVals]


def getRL(args):
    '''Get Ranking Loss ~ O(usrs * attributes ** 2)
    >>> args = {
    ... 'u2probs': {
    ...     'usr1': [0.1, 0.9, 0.2, 0.7, 0.1],
    ...     'usr2': [0.7, 0.3, 0.5, 0.2, 0.1]
    ...  },
    ... 'usr2NonzeroCols': {
    ... 'usr1': [0, 3],
    ... 'usr2': [1, 2],
    ... },
    ... 'totalLabelsNum': 5,
    ... 'rlPairsCnt': 6,
    ... }
    >>> print round(getRL(args), 3)
    0.417
    '''
    u2probs = args['u2probs']
    usr2NonzeroCols = args['usr2NonzeroCols']
    totalLabelsNum = args['totalLabelsNum']
    totalLabels = range(totalLabelsNum)
    rlPairsCnt = args['rlPairsCnt']

    errors = 0.0
    for usr in u2probs:
        onesAct = usr2NonzeroCols[usr]
        zerosAct = list(set(totalLabels[:]) - set(onesAct))
        allCombs = list(product(onesAct, zerosAct))
        probs = u2probs[usr]
        errorCnt = 0

        # Count err by each comb's prob order
        for comb in allCombs:
            onePos, zeroPos = comb[0], comb[1]
            errorCnt += 1 if probs[onePos] < probs[zeroPos] else 0

        # Add to error
        errors += errorCnt / float(rlPairsCnt)

    return errors / len(u2probs)


def getCoverage(args):
    '''Get Coverage Loss, O(usrs * mlogm), m = attributes
    >>> args = {
    ... 'u2probs': {
    ...     'usr1': [0.1, 0.9, 0.2, 0.7, 0.1],
    ...     'usr2': [0.7, 0.3, 0.5, 0.2, 0.1]
    ...  },
    ... 'usr2NonzeroCols': {
    ... 'usr1': [0, 3],
    ... 'usr2': [1, 2],
    ... },
    ... 'totalLabelsNum': 5,
    ... }
    >>> print round(getCoverage(args), 3)
    0.7
    '''
    u2probs = args['u2probs']
    usr2NonzeroCols = args['usr2NonzeroCols']
    totalLabelsNum = args['totalLabelsNum']

    errors = 0.0
    for usr in u2probs:
        probs = u2probs[usr]
        posSorted = getSortedPos(probs, reverse=True)  # Get sorted position by val (by probs)
        onesAct = usr2NonzeroCols[usr]

        # Collect this usr error by the last one's pos by stable-sort in probs
        error = (max([posSorted.index(pos) for pos in onesAct]) + 1) / float(totalLabelsNum)

        errors += error

    return errors / len(u2probs)


def getAvgPrecision(args):
    '''Get Avg Precision, O(usrs * mlogm), m = attributes
    >>> args = {
    ... 'u2probs': {
    ...     'usr1': [0.1, 0.9, 0.2, 0.7, 0.1],
    ...     'usr2': [0.7, 0.3, 0.5, 0.2, 0.1]
    ...  },
    ... 'usr2NonzeroCols': {
    ... 'usr1': [0, 3],
    ... 'usr2': [1, 2],
    ... },
    ... 'totalLabelsNum': 5,
    ... }
    >>> print round(getAvgPrecision(args), 3)
    0.542
    '''
    def getRanks(vals):
        # Rank from 1 to len(vals) for vals
        posVals = [(ind, val) for ind, val in enumerate(vals)]
        posVals.sort(key=lambda tup: tup[1])
        rankPos = [(ind + 1, tup[0]) for ind, tup in enumerate(posVals)]  # Rank starts from 1
        rankPos.sort(key=lambda tup: tup[1])
        return [tup[0] for tup in rankPos]

    u2probs = args['u2probs']
    usr2NonzeroCols = args['usr2NonzeroCols']

    errors = 0.0
    for usr in u2probs:
        onesAct = usr2NonzeroCols[usr]
        probs = u2probs[usr]
        posSorted = getSortedPos(probs, reverse=True)  # Get sorted position by val (by probs)

        # Get two ranks for each 'one'
        totalRanks4Ones = [posSorted.index(one) + 1 for one in onesAct]  # Rank starts from one
        onesRanks4Ones = getRanks(totalRanks4Ones)

        errors += sum([tup[0] / float(tup[1]) for tup in zip(onesRanks4Ones, totalRanks4Ones)]) / len(onesAct)

    return errors / len(u2probs)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
