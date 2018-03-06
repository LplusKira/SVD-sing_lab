from SNE_lab.dataloaders.MovieLens1M import ML1MLoader
from pandas import read_csv  # Load csv fast
from os import environ


class ML1MLoader(ML1MLoader):
    def __init__(self, rating_file='../data/ml-1m/ratings.dat.filtered', usr2labels_file='../data/ml-1m/users.dat.one.filtered', sub=None):
        self.SVD_K_NUM = int(environ.get('SVD_K_NUM', 1))
        super(ML1MLoader, self).__init__(rating_file, usr2labels_file, sub)

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
        ratings = records[:, 2].astype(float)
        return usrs, items, ratings

    def getLabelsCnt(self):
        return len(self.attr_bds)

    # Override getTrainingConf
    def getTrainingConf(self):
        return self.SVD_K_NUM, \
            self.MAX_TRAIN_NUM, \
            self.LEARNING_RATE, \
            self.LAMBDA
