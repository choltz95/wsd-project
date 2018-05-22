import numpy as np
import sys
sys.path.insert(0, '..')
from utils import *
import scipy.io as sio
import h5py
from scipy.sparse import csc_matrix


def load_matrices(filepath, Cfilename, Wfilename):
    #C = np.load(filepath + '/' + Cfilename) # global word-vector matrix
    C = sio.loadmat(filepath + '/' + Cfilename)['C']
    W = np.load(filepath + '/' + Wfilename) # cooc-based weight matrix

    return C,W
    
def compute_embeddings(qname, aname, vocab2id,C, W):
    dim = C.shape[0]
    fidq= open(qname,'r')
    nlines = int(fidq.readline())
    fida= open(aname,'w')
    fida.write('%d,%d\n' % (nlines,dim))
    
    for l in fidq:
        line = l.split(',')
        testnum = int(line[0])
        linenum = int(line[1])
        label = line[2].strip()
        word = line[3].strip()
        context = [line[i].strip() for i in xrange(4,len(line))] 
        
        wid = vocab2id[word] 
        cid = [vocab2id[c] for c in context]
     
        if len(cid) == 0:
            vec = np.zeros((dim,1))
            nrm = 0.0
            #print 'context is too small: %d' % len(cid)
            if len(cid) == 0: raise
        else:
            vec = np.dot(C[:,cid], W[cid,wid])
            
            nrm = np.linalg.norm(vec,2)
            nrm /= (np.sqrt(len(cid)+0.0)) 
        
        fida.write('%d,%d,%s,%e,' % (testnum,linenum,label,nrm))
        for x in vec:
            fida.write('%e,' % x)
        fida.write('\n')
        
    fida.close() 
    fidq.close()
    print 'done'
  
if __name__ == "__main__": 
    C, W = load_matrices('.','C_trainon_wiki_vocabwiki_dim50.mat','W.npy')
    vocab2id, freq = load_freq_vocabid('.')
    testtype = 'relevance'
    vocabtype = 'enwiki'
    sentences = 'tests/wordnet_%s_questions_%s_withoutword.txt' % (vocabtype,testtype)
    outfile = 'embeddings/embeddings_arora_%s.txt' % (vocabtype)
    compute_embeddings(sentences, outfile, vocab2id, C, W)
