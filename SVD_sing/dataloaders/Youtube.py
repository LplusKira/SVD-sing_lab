from SNE_lab.dataloaders.Youtube import YTLoader
from pandas import read_csv  # Load csv fast
from numpy import full
from os import environ


class YTLoader(YTLoader):
    def __init__(self, rating_file='../data/youtube/com-youtube.ungraph.txt.top10', usr2labels_file='../data/youtube/com-youtube.all.cmty.txt.top10.filtered', sub=None):
        self.SVD_K_NUM = int(environ.get('SVD_K_NUM', 1))
        super(YTLoader, self).__init__(rating_file, usr2labels_file, sub)

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
