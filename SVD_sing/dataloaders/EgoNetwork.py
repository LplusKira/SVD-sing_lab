from SNE_lab.dataloaders.EgoNetwork import ENLoader
from pandas import read_csv  # Load csv fast
from numpy import full
from os import environ


class ENLoader(ENLoader):
    def __init__(self, rating_file='../data/ego-net/3980.edges.u2u', usr2labels_file='../data/ego-net/3980.circles.u2f.filtered', sub='3980', silence=False):
        self.SVD_K_NUM = int(environ.get('SVD_K_NUM', 1))
        super(ENLoader, self).__init__(rating_file, usr2labels_file, sub, silence)

    # Overwrite load
    def load(self):
        '''
        sample:
            each line (from input) id, r1, r2, ...., rn
              e.g. 123, 1, 0, 3
            records = array([[ 1.,  0.,  2.],
                             [ 0.,  2.,  3.]])
            ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html
        '''
        records = read_csv(self.rating_file, sep=',', header=None).as_matrix()
        usrs = records[:, 0]
        items = records[:, 1]
        ratings = full(usrs.shape, 3, dtype=float)  # init with 3; ego-net's data dont have 'ratings'
        return usrs, items, ratings

    def getLabelsCnt(self):
        return len(self.attr_bds)

    # Override getTrainingConf
    def getTrainingConf(self):
        return self.SVD_K_NUM, \
            self.MAX_TRAIN_NUM, \
            self.LEARNING_RATE, \
            self.LAMBDA
