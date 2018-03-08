import logging
from os import environ

# For learning
THRESH = float(environ.get('THRESH', 1e-12))

# For logging
DEBUG2LOG_LEVEL = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}
DEBUG = environ.get('DEBUG')  # Logging level
LOG_LEVEL = DEBUG2LOG_LEVEL.get(DEBUG, DEBUG2LOG_LEVEL['INFO'])

# FOr test (indep from logging level)
TEST_SVD_SING = bool(environ.get('TEST_SVD_SING'))
TEST_SNE = TEST_SVD_SING
TEST_DIR = 'report/.testsvdsing/'
