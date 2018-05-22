import os
import sys
import logging
import itertools
from collections import defaultdict

import numpy
import scipy.sparse

# parameters controlling what is to be computed: how many dimensions, window size etc.
DIM = 600
DOC_LIMIT = None  # None for no limit
TOKEN_LIMIT = 30000
WORKERS = 8
WINDOW = 10
DYNAMIC_WINDOW = False
NEGATIVE = 10  # 0 for plain hierarchical softmax (no negative sampling)

logger = logging.getLogger("run_embed")

from cooccur_matrix import get_cooccur


