import os
import numpy as np
from gensim.models import word2vec
import gensim.downloader as api
from gensim.matutils import corpus2csc
from gensim.corpora import Dictionary
from gensim.utils import SaveLoad

from tqdm import tqdm

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# parameters controlling what is to be computed: how many dimensions, window size etc.
WORKERS = 8
WINDOW = 10
DYNAMIC_WINDOW = False

dataset = api.load("text8")
dct = Dictionary(dataset)  # fit dictionary
word2id = dct.token2id
#bow_corpus = [dct.doc2bow(line) for line in dataset]
#term_doc_mat = corpus2csc(bow_corpus,dtype=np.int16,printprogress=200)
#term_term_mat = np.dot(term_doc_mat, term_doc_mat.T)
#np.save('cooc.npy', term_term_mat)

# filter sentences to contain only the dictionary words
corpus = lambda: ([word for word in sentence if word in word2id] for sentence in dataset)
#import glove  # https://github.com/maciejkula/glove-python
#print('training glove...')
#cooccur = glove.Corpus(dictionary=word2id)
#cooccur.fit(corpus(), window=WINDOW)

#utils.pickle(cooccur, outf('glove_corpus'))
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
from cooccur_matrix import get_cooccur
outf = lambda prefix: os.path.join(sys.argv[3], prefix)
print('getting cooccur')
raw = get_cooccur(corpus(), word2id, window=WINDOW, dynamic_window=False)
print('saving cooccur')
np.save('cooccur.npy', raw)
print('done')
