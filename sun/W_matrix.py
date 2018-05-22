import os
import numpy as np
from gensim.models import word2vec
import gensim.downloader as api
from gensim.matutils import corpus2dense
from gensim.corpora import Dictionary
from gensim.utils import SaveLoad
import pandas as pd
from tqdm import tqdm
from utils import *

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# parameters controlling what is to be computed: how many dimensions, window size etc.
WORKERS = 8
WINDOW = 10
DYNAMIC_WINDOW = False

dataset = api.load("text8")
dct = Dictionary(dataset,prune_at=20000)  # fit dictionary
dct.filter_extremes(no_below=10,no_above=0.5, keep_n=20000)
dct.compactify()

#word2id2 = dct.token2id
#word2idf = open('enwiki_wordID2000.csv').read().splitlines()
#word2id = dict(map(lambda s : map(str.strip, s.split(',')), word2idf))
#word2id = dict((k,int(v)) for k,v in word2id.items())
word2id, freq = load_freq_vocabid('.')
# filter sentences to contain only the dictionary words
corpus = lambda: ([word for word in sentence if word in word2id] for sentence in dataset)

import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
from cooccur_matrix import get_cooccur
outf = lambda prefix: os.path.join(sys.argv[3], prefix)
print('getting cooccur')
raw = get_cooccur(corpus(), word2id, window=WINDOW, dynamic_window=False)

np.save('W.npy', raw)
print('done')
