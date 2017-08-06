import os

# modify by data format
USR_TOTAL_LABELS_FIELDS = int(os.environ.get('USR_TOTAL_LABELS_FIELDS')) if('USR_TOTAL_LABELS_FIELDS' in os.environ) else (2 + 4 + 2 + 2 + 2) # age/occupation/gender

# for training
SVD_K_NUM       = int(os.environ.get('SVD_K_NUM')) if('SVD_K_NUM' in os.environ) else 10
print '[info] SVD_K_NUM: ', SVD_K_NUM

MAX_TRAIN_NUM   = int(os.environ.get('MAX_TRAIN_NUM')) if('MAX_TRAIN_NUM' in os.environ) else 100
LEARNING_RATE   = float(os.environ.get('LEARNING_RATE')) if('LEARNING_RATE' in os.environ) else 0.0001 # update = (1 *             OR          / MOMENTUM) * LEARNING_RATE * gradient
LAMBDA          = float(os.environ.get('LAMBDA')) if('LAMBDA' in os.environ) else 0.001

# for debugging
DEBUG = int(os.environ.get('DEBUG')) if('DEBUG' in os.environ) else 0
