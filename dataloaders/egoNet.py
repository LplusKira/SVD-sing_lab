import sys
import traceback
from bisect import bisect_left
sys.path.insert(0, '../')

class dataloader:
    def __init__(self):
        pass

    ## get each, in usrs, usr's labels 
    def get_labels(self, usr2labels_file, usrs):
        usr2labels = {}
        usr2nonZeroCols = {}
        fd = open(usr2labels_file, 'r')
        for line in fd:
            try:
                line = line.strip().split(',')
                usr = int(line[0])

                if usr in usrs:
                    # get formaulated labels
                    usr2labels[usr] = [int(e) for i, e in enumerate(line[1:])] 
            except:
                print traceback.format_exc()
                raise
        fd.close()
        return usr2labels

    # not done with get_labels to avoid 'too many values to unpack'
    def get_nonZeroCols(self, usr2labels_file):
        usr2labels = {}
        usr2nonZeroCols = {}
        fd = open(usr2labels_file, 'r')
        for line in fd:
            try:
                line = line.strip().split(',')
                usr = int(line[0])

                usr2nonZeroCols[usr] = [i for i, e in enumerate(line[1:]) if int(e) != 0] 
            except:
                print traceback.format_exc()
                raise
        fd.close()
        return usr2nonZeroCols

