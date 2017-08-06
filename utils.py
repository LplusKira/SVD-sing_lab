import sys
import traceback
import random
from config import DEBUG
from bisect import bisect_left
random.seed(87)

def debug(msg, val):
    if DEBUG > 0:
        print '[info] ' + msg, val

