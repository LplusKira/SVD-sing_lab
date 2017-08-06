import dataloaders.yelp as qqdataloader                
from utils import debug
from config import USR_TOTAL_LABELS_FIELDS, MAX_TRAIN_NUM, LAMBDA, LEARNING_RATE, SVD_K_NUM

import random, sys, traceback, scipy
import numpy as np
from time import gmtime, strftime
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn import linear_model

np.random.seed( int(sys.argv[3]) ) # Reproducibility
random.seed( int(sys.argv[3]) )    # Reproducibility

def printTruePredicted(u2predictions, usr2NonzeroCols):
    for usrid in u2predictions:
        print usrid, usr2NonzeroCols[usrid], u2predictions[usrid]

# return {
#  0: tp list(of cols) 
#  1: fp list(of cols)
#  2: tn list(of cols)
# }
def getClasses(trueCols, predictedCols):
    classDict = {
      0: [],
      1: [],
      2: [],
    }
    for col in trueCols:
        if col in predictedCols:
            classDict[0].append(col)
        else:
            classDict[2].append(col)
    for col in predictedCols:
        if col not in trueCols:
            classDict[1].append(col)
    return classDict

# ref(how to cal microf1): http://rushdishams.blogspot.tw/2011/08/micro-and-macro-average-of-precision.html
# get micro f1 by age/gender/occupation
def getMicroF1ByCol(u2predictions, usr2NonzeroCols):
    tpList = [0.0] * USR_TOTAL_LABELS_FIELDS
    fpList = [0.0] * USR_TOTAL_LABELS_FIELDS
    tnList = [0.0] * USR_TOTAL_LABELS_FIELDS
    for usrid in u2predictions:
        y_nonzeroCols = usr2NonzeroCols[usrid] # actual
        bestCols = u2predictions[usrid]        # predicted

        # update tp, fp, tn
        # 0: tp, 1: fp, 2: tn
        classDict = getClasses(y_nonzeroCols, bestCols)
        for col in classDict[0]:
            tpList[col] += 1.0
        for col in classDict[1]:
            fpList[col] += 1.0
        for col in classDict[2]:
            tnList[col] += 1.0
    
    # cal micro precision & recall    
    #   micor precision = sum(tp) / (sum(tp) + sum(fp))
    #   micro recall    = sum(tp) / (sum(tp) + sum(tn))
    summedTp = sum(tpList)
    microPrecision = summedTp/ (summedTp + sum(fpList))
    microRecall    = summedTp/ (summedTp + sum(tnList))

    # cal micro F1
    microF1        = 2 * microPrecision * microRecall / (microPrecision + microRecall) if(summedTp > 0) else 0.0
    return microF1

# get one error
#   one error = sum( has one class hits or not ) / dataPointsNum
def getOneError(u2predictions, usr2NonzeroCols):
    errCnt = len(u2predictions)
    usrCnt = len(u2predictions)
    for usrid in u2predictions:
        y_nonzeroCols = usr2NonzeroCols[usrid]
        bestCols = u2predictions[usrid]
        for ind, col in enumerate(bestCols):
            if col == y_nonzeroCols[ind]:
                # if one class(col) hits, then no err for this usr
                errCnt -= 1
                break
    return errCnt / float(usrCnt)
                
# get RL (ranking loss)
#   it's .. (0,1) pair's examination 
def getRL(u2predictions, usr2NonzeroCols):
    colNums = USR_TOTAL_LABELS_FIELDS * (USR_TOTAL_LABELS_FIELDS - len(usr2NonzeroCols.itervalues().next()))
    totalLoss = 0.0
    for usrid in u2predictions:
        y_nonzeroCols = usr2NonzeroCols[usrid]
        y_zeroCols = range(USR_TOTAL_LABELS_FIELDS)
        col2Val = map(lambda v: [v, 0], y_zeroCols)
        for v in y_nonzeroCols:
            y_zeroCols.remove(v)

        # get the most possible cols' combination + add to sort list
        bestCols = u2predictions[usrid]
        for v in bestCols:
            col2Val[v][1] = 1
        col2Val.sort(key=lambda v: v[1], reverse=True)
        col2Order = {}
        for ind, v in enumerate(col2Val):
            col2Order[v[0]] = ind
        
        # check for every true's '0','1''s indx pair
        # if the ordery of predicted is reverse => err ++
        errCnt = 0
        for col1 in y_zeroCols:
            for col2 in y_nonzeroCols:
                if col2Order[col1] < col2Order[col2]:
                    errCnt += 1
        lossPerUsr = errCnt / colNums 
        totalLoss += lossPerUsr

    return totalLoss / len(u2predictions)
        
# get coverage 
#   covreage = find the last one's position (ranked by predicted probability)
#   we may assume 0 1 | 1 0 0:   prob pos(1) >= prob pos(0); prob pos(2) >= prob pos(3), prob pos(2) >= prob pos(4),
#             pos 0 1   2 3 4 
#   but have no other knowledge, so 'sort by prob' would just have: 1 1 0 0 0 (i.e. doesnt change its original ordery)
#                                                                   1 2 0 3 4
def getCoverage(u2predictions, usr2NonzeroCols):
    totalFields = 1.0 * USR_TOTAL_LABELS_FIELDS
    colNums = len( next(usr2NonzeroCols.itervalues()) )
    loss = 0.0
    for usrid in u2predictions:
        y_nonzeroCols = usr2NonzeroCols[usrid]
        bestCols = u2predictions[usrid]
           
        # rank by prob (start from 0): i.e. lowest prob => bigger rank number
        lowestOneRank = colNums - 1
        for cnt in range(0, colNums):
            ind = colNums - 1 - cnt
            if bestCols[ind] > y_nonzeroCols[ind]:
                lowestOneRank = y_nonzeroCols[ind] + cnt + 1
                break
            elif bestCols[ind] < y_nonzeroCols[ind]:
                lowestOneRank = y_nonzeroCols[ind] + cnt
                break
            
        loss += lowestOneRank / totalFields
    return loss / len(usr2itemsIndx)


# get average precision 
#   we may assume 0 1 | 1 0 0:   prob pos(1) >= prob pos(0); prob pos(2) >= prob pos(3), prob pos(2) >= prob pos(4),
#             pos 0 1   2 3 4
#   since we still dont have each field's prob
#     so 'sort by prob' would just have: 1 1 0 0 0 (i.e. doesnt change its original ordery)
#                                        1 2 0 3 4
def getAvgPrecision(u2predictions, usr2NonzeroCols):
    colNums = len( next(usr2NonzeroCols.itervalues()) )
    prec = 0.0
    for usrid in u2predictions:
        y_nonzeroCols = usr2NonzeroCols[usrid]
        bestCols = u2predictions[usrid]
           
        # each 'real one' has a value: (its reverse pos + 1 in 'ones' by prob) / (its reverse pos + 1 in all fields by prob)
        #                              ^^ i.e. higher porb has lowe pos
        col2AllRank = {}
        score = 0.0 
        for cnt in range(0, colNums):
            ind = colNums - 1 - cnt
            if y_nonzeroCols[ind] == bestCols[ind]:
              col2AllRank[ y_nonzeroCols[ind] ] = ind + 1
            else:
              col = y_nonzeroCols[ind]
              col2AllRank[ col ] = col + len(filter(lambda v: v > col, bestCols))  + 1

        # sort by Allrank lower to bigger 
        rankedList = sorted(col2AllRank.items(), key=lambda x: x[1])
        for ind, val in enumerate(rankedList):
          score += float(ind + 1) / val[1]

        prec += score / colNums
                
    return prec / len(usr2itemsIndx)

# get  hamming loss
#   we may assume 0 1 | 1 0 0:  pred 
#                 1 0 | 0 1 0:  real 
#   since the papaer itself doesnt specify, we use pred XOR(by attribute) real (i.e. 1 | 1 => hamming loss (for this): 2/2)
def getHammingLoss(u2predictions, usr2NonzeroCols):
    colNums = len( next(usr2NonzeroCols.itervalues()) )
    loss = 0.0
    for usrid in u2predictions:
        y_nonzeroCols = usr2NonzeroCols[usrid]
        bestCols = u2predictions[usrid]
           
        dataPointLoss = 0.0
        for ind, val in enumerate(bestCols):
            if bestCols[ind] != y_nonzeroCols[ind]:
                dataPointLoss += 1.0
    
        loss += dataPointLoss / colNums
    return loss / len(u2predictions)


# pick 10% as valid data
def splitTrainTest(usrs, items, ratings):
    uniqUsrs = np.unique(usrs)
    validUsrs = random.sample(uniqUsrs, int(uniqUsrs.shape[0] * 0.1))
    validIndx = [ind for ind in range(0, usrs.shape[0]) if usrs[ind] in validUsrs]
    trainIndx = [ind for ind in range(0, usrs.shape[0]) if ind not in validIndx]

    return ratings[trainIndx], ratings[validIndx], usrs[trainIndx], usrs[validIndx], items[trainIndx], items[validIndx]
  
def main(argv):
    if not len(argv) == 3:
        print '[info] usage: python run.py yourtraindata yourlabelData randomSeed(int)'
        return 1
    
    ''' load each usr's ratings (train/valid data) ''' 
    print '\n[info] time == ', strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print "[info] hi im loading from", argv[0] 
    # sample: 
    # each line (from input) id, r1, r2, ...., rn
    #   e.g. 123, 1, 0, 3
    # ratings = array([[ 1.,  0.,  2.],
    #                  [ 0.,  2.,  3.]])
    # ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html
    records = np.loadtxt(argv[0], dtype=float, delimiter='\t')
    usrs = records[:,0]
    items = records[:,1]
    ratings = records[:,2]
    ratingsTrain, ratingsValid, usrsTrain, usrsValid, itemsTrain, itemsValid = splitTrainTest(usrs, items, ratings)
    uniqUsrsTrainList = np.unique(usrsTrain).tolist()
    uniqUsrsValidList = np.unique(usrsValid).tolist()
    train4svd = scipy.sparse.csc_matrix((ratingsTrain, ([uniqUsrsTrainList.index(usr) for usr in usrsTrain], itemsTrain)))
    valid4svd = scipy.sparse.csc_matrix((ratingsValid, ([uniqUsrsValidList.index(usr) for usr in usrsValid], itemsValid)))
    print '[info] ratingsTrain, ratingsValid loaded; train4svd, valid4svd loaded'
    print '[info] usrs in train: ', uniqUsrsTrainList
    print '[info] usrs in valid: ', uniqUsrsValidList
    
    
    ''' acquire (for all usrs) usr2labels & usr2NonzeroCols ''' 
    print '\n[info] time == ', strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print "[info] labels are loading from", argv[1] 
    # sample:
    # each line (from input) id,on-hot-encoded labels 
    #   e.g. 123,0,1,1,0,0
    # usr2labels = {
    #   0: [0,0,1, 1,0],
    #   1: [1,0,0, 0,1],
    # }
    # usr2NonzeroCols = {
    #   0: [2, 3],
    #   1: [0, 4],
    # }
    dataloader = qqdataloader.dataloader()
    usr2labels = dataloader.get_labels(argv[1], usrs)        
    usr2NonzeroCols = dataloader.get_nonZeroCols(argv[1])
    print '[info] usr2labels, usr2NonzeroCols loaded'


    ''' svd, i.e. u * s * vt '''
    # ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html
    uTrain, sTrain, vtTrain = svds(train4svd, SVD_K_NUM, which = 'LM')
    uValid, sValid, vtValid = svds(valid4svd, SVD_K_NUM, which = 'LM')
    print '[info] shapes before svd: train valid', train4svd.shape, valid4svd.shape
    print '[info] svd done'


    ''' learning for W in logistic regression for each attr'''
    # ref: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    u2predictionsTrain = dict((el,[]) for el in uniqUsrsTrainList)
    u2predictionsValid = dict((el,[]) for el in uniqUsrsValidList)
    logreg = linear_model.LogisticRegression(C=LAMBDA, max_iter=MAX_TRAIN_NUM) #, verbose=1)
    for attrInd in range(0, len(usr2NonzeroCols[int( uniqUsrsTrainList[0] )])):
        for u in usr2NonzeroCols:
            if not (len(usr2NonzeroCols[u]) == 3):
                print u, usr2NonzeroCols[u]
        trainAttrs = [usr2NonzeroCols[int(usr)][attrInd] for usr in uniqUsrsTrainList]
        validAttrs = [usr2NonzeroCols[int(usr)][attrInd] for usr in uniqUsrsValidList]
        logreg.fit(uTrain, trainAttrs)

        predictionsTrain = logreg.predict(uTrain)
        predictionsValid = logreg.predict(uValid)
        #print '\non attrind: ', attrInd
        #print 'on train: actual, predict', trainAttrs, logreg.predict(uTrain)
        #print 'on train: usrs', uniqUsrsTrainList

        # append this attr's prediction by usr
        for ind in range(0, uTrain.shape[0]):
            usrID = uniqUsrsTrainList[ind]
            u2predictionsTrain[ usrID ].append( predictionsTrain[ind] )
        for ind in range(0, uValid.shape[0]):
            usrID = uniqUsrsValidList[ind]
            u2predictionsValid[ usrID ].append( predictionsValid[ind] )

    print '\n[info] time == ', strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print '[info] logistic regression done'

    microF1Train = getMicroF1ByCol(u2predictionsTrain, usr2NonzeroCols)
    microF1Valid = getMicroF1ByCol(u2predictionsValid, usr2NonzeroCols)
    oneErrorTrain = getOneError(u2predictionsTrain, usr2NonzeroCols)
    oneErrorValid = getOneError(u2predictionsValid, usr2NonzeroCols)
    #RLTrain = getRL(u2predictionsTrain, usr2NonzeroCols)
    #RLValid = getRL(u2predictionsValid, usr2NonzeroCols)
    #coverageTrain = getCoverage(u2predictionsTrain, usr2NonzeroCols)
    #coverageValid = getCoverage(u2predictionsValid, usr2NonzeroCols)
    #avgPrecTrain = getAvgPrecision(u2predictionsTrain, usr2NonzeroCols)
    #avgPrecValid = getAvgPrecision(u2predictionsValid, usr2NonzeroCols)
    HLTrain = getHammingLoss(u2predictionsTrain, usr2NonzeroCols)
    HLValid = getHammingLoss(u2predictionsValid, usr2NonzeroCols)
    print '[info] train data microF1 == ', microF1Train
    print '[info] valid data microF1 == ', microF1Valid
    print '[info] train data oneError == ', oneErrorTrain
    print '[info] valid data oneError == ', oneErrorValid
    #print '[info] train data RL == ', RLTrain
    #print '[info] valid data RL == ', RLValid
    #print '[info] train data coverage == ', coverageTrain
    #print '[info] valid data coverage == ', coverageValid
    #print '[info] train data avgPrec == ', avgPrecTrain
    #print '[info] valid data avgPrec == ', avgPrecValid
    print '[info] train data hammingLoss == ', HLTrain
    print '[info] valid data hammingLoss == ', HLValid
    print '\n[info] time == ', strftime("%Y-%m-%d %H:%M:%S", gmtime())

    print '[info]: for traindata, print real vals & predicted vals ... '
    printTruePredicted(u2predictionsTrain, usr2NonzeroCols)
    print '[info]: for validdata, print real vals & predicted vals ... '
    printTruePredicted(u2predictionsValid, usr2NonzeroCols)

if __name__ == '__main__':
    main(sys.argv[1:])
