import numpy as np
import scipy.sparse as ss
import os.path
import scipy.io as sio
from utils import *


"""
Load all of embeddings in one database
"""

def get_embeddings(filename,vocab2id):
    fid = open(filename,'r')
    line = fid.readline().split(' ')
    
    
    vectorlist = [None for x in xrange(len(vocab2id))]
    while(1):
        line = fid.readline().split()
        if len(line) == 0: break
        word = line[0]
    
        ns = int(line[1].strip('\n'))
        
        gvec = fid.readline().split()
        gvec = [float(g.strip('\n'))  for g in gvec]
        d = {'global':gvec,'sense':[],'center':[],'word':word}
       
        for k in xrange(ns):
            cl = fid.readline().split()
            cl = [float(c.strip('\n'))  for c in cl]
            
            cc = fid.readline().split()
            cc = [float(c.strip('\n'))  for c in cc]
                  
            d['sense'].append(cl)
            d['center'].append(cc)
        if word in vocab2id:
            i = vocab2id[word]
            vectorlist[i] = d        
    fid.close()
    return vectorlist
    
    


"""
Function to transfer question (words) to answers (list of multisense embeddings)
"""

def compute_question2embed(qname, aname,vocab2id, vectorlist,disttype):
    
    fidq = open(qname,'r')
    nlines = int(fidq.readline())
    
    fida = open(aname,'w')
    fida.write('%d,%d\n' % (nlines,0))
    
    for l in fidq:
        line = l.split(',')
        
        testnum = int(line[0])
        linenum = int(line[1])
        label = line[2].strip()
        word = line[3].strip()
        context = [line[i].strip() for i in xrange(4,len(line))]
        
                   
        cvec = None
        relcontext = 0.0
        for c in context:
            
            i = vocab2id[c]
            if  vectorlist[i] is None:
                continue
            relcontext += 1.0
            vg = np.array(vectorlist[i]['global'])
            if cvec is None: cvec = vg
            else: cvec += vg
                
        if relcontext == 0.0: continue
        cvec /= relcontext
        
        
        i = vocab2id[word]
        if vectorlist[i] is None:
            print "word %s not in vocab" % word
            continue
        
        d = []
        for c in vectorlist[i]['center']:
            d.append(get_dist(c,cvec,1.0))
                
        j = np.argmax(np.array(d))
        vec = vectorlist[i]['sense'][j]
        nrm = d[j]

        fida.write('%d,%d,%s,%f,' % (testnum,linenum,label,nrm))
        for x in vec:
            fida.write('%f,' % x)
        fida.write('\n')

    fida.close()
        
    fidq.close()
    print 'done'

if __name__ == "__main__":
       
    vocabtype = 'enwiki'
    
    vocab2id,freq = load_freq_vocabid('..') 
    V = len(vocab2id)
    
    filename = 'chen_embeds_in_amherestformat.txt'
    
    vectorlist = get_embeddings(filename, vocab2id)
    print 'embedding loaded'
    

    tail =  'maxsense_%s' % disttype
    print tail

    testclass = 'wordnet'
    testtype = 'relevance'

    qname = '../tests/wordnet_%s_questions_%s_withoutword.txt' % (vocabtype,testtype)  
    aname = 'embeddings/embeddings_chen_%s_%s_%s_nowordincontext_%s.txt' % (vocabtype,testclass,testtype,tail)
    compute_question2embed(qname, aname, vocab2id, vectorlist,disttype)
