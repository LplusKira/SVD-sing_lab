import sys
import random
import logging
from numpy import random as nprandom
from numpy import unique
from scipy.sparse.linalg import svds  # For sparse matrix SVD
from sklearn import linear_model  # For Logistic Reg

from config import LOG_LEVEL, TEST_SVD_SING
from utils import splitKfolds, getTrainValid, loadToCSCMatrix, getDataStats, handlePrediction
from dataloaders.EgoNetwork import ENLoader
from dataloaders.Youtube import YTLoader
from dataloaders.MovieLens100K import ML100KLoader
from dataloaders.MovieLens1M import ML1MLoader
from SNE_lab.utils import getMicroF1ByCol, getOneError, getRL, getCoverage, getAvgPrecision, getHammingLoss
from SNE_lab.statevalidators.ENValidator import ENValidator
from SNE_lab.statevalidators.YTValidator import YTValidator
from SNE_lab.statevalidators.ML1MValidator import ML1MValidator
from SNE_lab.statevalidators.ML100KValidator import ML100KValidator
DATA2LOADER = {  # By subdir in data/
    'ml-100k': ML100KLoader,
    'ml-1m': ML1MLoader,
    'ego-net': ENLoader,
    'youtube': YTLoader,
}
DATA2VALIDATOR = {  # By subdir in data/
    'ml-100k': ML100KValidator,
    'ml-1m': ML1MValidator,
    'ego-net': ENValidator,
    'youtube': YTValidator,
}


def printTruePredicted(u2predictions, u2probs, usr2NonzeroCols):
    ROUND_DIGITS = 3
    for usr in u2predictions:
        roundedProbs = [round(prob, ROUND_DIGITS) for prob in u2probs[usr]]
        print(' '.join([str(usr), str(usr2NonzeroCols[usr]), str(u2predictions[usr]), str(roundedProbs)]))


def parseArgs(argv, **kwargs):
    def getSubDir(rating_file, usr2labels_file):
        rating_file_subdir = rating_file.split('/')[2]
        usr2labels_file_subdir = usr2labels_file.split('/')[2]
        if rating_file_subdir == usr2labels_file_subdir:
            return rating_file_subdir
        return None

    mainFile = argv[0]
    nprandom.seed(int(argv[1]))  # Reproducibility
    random.seed(int(argv[1]))  # Reproducibility
    foldNum = int(argv[2])
    dataset = argv[3]
    rating_file = kwargs.get('rating_file')  # By 'get', default None
    usr2labels_file = kwargs.get('usr2labels_file')
    subtitle = kwargs.get('sub')

    usage = '[USEAGE] python -u ' + mainFile + \
        ' randomSeed(int) foldNum dataset subtitle(opt) ratingFile(opt) featureFile(opt)' + \
        '(ratingFile and featureFile only coexist under the same dir)'
    # Handle ratingFile, featureFile, dataset's conflicts
    cond1 = not rating_file and usr2labels_file
    cond2 = rating_file and not usr2labels_file
    cond3 = rating_file and usr2labels_file
    if cond1 or cond2:
        raise Exception(usage)
    elif cond3:
        subdir = getSubDir(rating_file, usr2labels_file)
        if not subdir:
            raise Exception(usage)
    return foldNum, dataset, subtitle, rating_file, usr2labels_file


def main(argv):
    foldNum, dataset, subtitle, rating_file, usr2labels_file = parseArgs(
        argv[:4],
        **dict(arg.split('=') for arg in argv[4:]))
    if rating_file and usr2labels_file:
        dataloader = DATA2LOADER[dataset](
            ratig_file=rating_file,
            usr2labels_file=usr2labels_file,
            sub=subtitle,
        )
    else:
        dataloader = DATA2LOADER[dataset]()

    '''Load training configs
    '''
    SVD_K_NUM, \
        MAX_TRAIN_NUM, \
        LEARNING_RATE, \
        LAMBDA = dataloader.getTrainingConf()

    '''Load usrs, items, ratings
    '''
    usrs, items, ratings = dataloader.load()
    uniqUsrs = unique(usrs)

    '''Acquire (for all usrs) usr2labels & usr2NonzeroCols
    '''
    usr2labels, usr2NonzeroCols = dataloader.get_labels(uniqUsrs)

    ''' K-fold validation
    '''
    kfolds = splitKfolds(len(uniqUsrs), foldNum, random.shuffle)
    for ind in range(foldNum):
        '''Get fold's train/valid data
        '''
        ratingsTrain, ratingsValid, \
            usrs4Train, usrs4Valid, \
            itemsTrain, itemsValid = getTrainValid(kfolds, ind, uniqUsrs, items, ratings)
        logging.info('{} usrs in train'.format(len(usrs4Train)))
        logging.info('{} usrs in valid'.format(len(usrs4Valid)))

        '''Init statevalidator
        '''
        statevalidator = DATA2VALIDATOR[dataset](
            dataset=dataset,
            datasetSub=dataloader.getDataSub(),
            curFold=ind,
            totalFolds=foldNum,
            usr2itemsIndxTrain=None,
            usr2itemsIndxValid=None,
            MAX_TRAIN_NUM=MAX_TRAIN_NUM,
            ITEM_FIELDS_NUM=SVD_K_NUM,
        )

        '''Load to scipy.sparse.csc_matrix (slow!)
        '''
        usrsShuffled = usrs4Train + usrs4Valid
        all4svd = loadToCSCMatrix(
            ratings=ratings,
            ratingUsrs=usrs,
            uniqShuffledUsrs=usrsShuffled,
            items=items,
        )
        logging.info('csc_matrix loaded')

        '''SVD -- decompose to u * s * vt
        ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html
        '''
        uAll, sAll, vtAll = svds(all4svd, SVD_K_NUM, which='LM')
        trainUsrsCnt = len(usrs4Train)
        uTrain = uAll[:trainUsrsCnt, :]
        uValid = uAll[trainUsrsCnt:, :]
        logging.info('usrs cnt, items cnt: {}'.format(all4svd.shape))
        logging.info('U dimensions = {}'.format(str(uAll.shape)))
        logging.info('S dimensions = {}'.format(str(sAll.shape)))
        logging.info('V\' dimensions = {}'.format(str(vtAll.shape)))
        logging.info('svd done')

        '''Init dataStats & logistic regression module
        '''
        dataStats = getDataStats(usrs4Train, usrs4Valid)
        logreg = linear_model.LogisticRegression(
            penalty='l2',
            dual=False,  # XXX
            tol=1e-12,
            C=1 / LAMBDA,
            fit_intercept=True,  # XXX
            intercept_scaling=1,  # XXX
            class_weight=None,
            random_state=None,  # None, rng same as np.random's
            solver='sag',  # For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs' handle multinomial loss; 'liblinear' is limited to one-versus-rest schemes.
            max_iter=MAX_TRAIN_NUM,
            multi_class='multinomial',  # Default: 'ovr'
            verbose=0,
            warm_start=False,  # wtf?
            n_jobs=1,  # Usefull only for multi_class: 'ovr'
        )

        '''Learning logistic regression's W by each attribute
        ref: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        '''
        attrsCnt = dataloader.getLabelsCnt()
        for ind in range(attrsCnt):
            # Train
            attrs = [usr2NonzeroCols[usr][ind] for usr in usrs4Train]
            logreg.fit(uTrain, attrs)

            # Handle prediction
            handlePrediction(dataStats, uTrain, uValid, logreg)
        logging.info('logistic regression done')

        # Collect Stats
        KPI2getters = {
            'microF1': getMicroF1ByCol,
            'oneError': getOneError,
            'RL': getRL,
            'coverage': getCoverage,
            'avgPrec': getAvgPrecision,
            'hammingLoss': getHammingLoss,
        }
        for d in dataStats:
            KPIArgs = {
                'usr2NonzeroCols': usr2NonzeroCols,
                'u2predictions': d['u2predictions'],
                'totalLabelsNum': dataloader.gettotalLabelsNum(),
                'rlPairsCnt': dataloader.getRLPairsCnt(),
            }
            d['KPIs'] = {kpi: getter(KPIArgs) for kpi, getter in KPI2getters.iteritems()}
            # OR (no write): statevalidator.logStats(d)
            statevalidator.writeCSVStats(d)

        # Log real, predicted
        if not TEST_SVD_SING:
            for d in dataStats:
                logging.info('for {}, print real & predicted & predicted probs'.format(d['name']))
                logging.info('usrid, actual, predicted, prob')
                printTruePredicted(d['u2predictions'], d['u2probs'], usr2NonzeroCols)

    return 1


if __name__ == '__main__':
    '''Set up logger and run training, validating, and recordeing stats
    '''
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=LOG_LEVEL)
    main(sys.argv[:])
