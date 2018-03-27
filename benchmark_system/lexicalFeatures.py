#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:28:34 2018

@author: xuecho
"""

from nltk.tokenize import TweetTokenizer
from collections import Counter
#from libsvm.svmutil import *
#from sklearn.svm import libsvm
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
import numpy as np
import logging
import re

logging.basicConfig(level=logging.INFO)


def parse_dataset(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with open(fp, 'rt', encoding='utf8') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                label = int(line.split("\t")[1])
                tweet = line.split("\t")[2]
                y.append(label)
                corpus.append(tweet)

    return corpus, y


def featurize(corpus):
    '''
    Tokenizes and creates TF-IDF BoW vectors with ngram_range=(1,4)
    :param corpus: A list of strings each string representing document.
    :return: X: A sparse csr matrix of TFIDF-weigted ngram counts.
    '''

    #tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    #vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words="english")
    tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=True).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words="english",ngram_range=(1,4))
    vectorizer.fit(corpus) # build ngram dictionary
    #ngram = vectorizer.fit_transform(corpus) # get ngram
    #print('ngram: {0}\n'.format(ngram))
    #print('ngram.shape: {0}'.format(ngram.shape))
    #print('vectorizer.vocabulary_: {0}'.format(vectorizer.vocabulary_))
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()
    # print(vectorizer.get_feature_names()) # to manually check if the tokens are reasonable
    
    return X

 #a set of numeric and binary features were included containing information about 

    #punctuation
    #capitalisation
    #hash-tag frequency ?
    #the hashtag-to-word ratio ?
    #(vii) emoticon frequency ?
    #(viii) tweet length ?
    #Where relevant, numeric features were normalised by dividing them by the tweet length in tokens.

def punctuation(corpus):
    punc = []
    ellips = []
    punclist =[]
    pat2 = '[(\.\.\.)]+'
    
    pattern = '(http.+(\s)?)|((@\w*\d*(\s)?))'
    corp1 = []
    for str in corpus:
        str1 = re.sub(pattern, '', str, flags=re.MULTILINE)
        corp1.append(str1)
    
    for a in corp1:
        #print(a)
        punc.append(a.count('!')+a.count('?')+a.count(','))
        ellips.append(len(re.findall(pat2,a)))
        t =[]
        t.append(a.count('!')+a.count('?')+a.count(','))
        t.append(len(re.findall(pat2,a)))
        punclist.append(t)
        
    #with open('punctuation.txt', 'w') as data_in:
        #for i in range(0,len(corpus)):
            #data_in.write("%f\t%f\n" %(punc[i],ellips[i]))
            
    return punc,ellips,punclist

def capitalisation(corpus):
    capitn = []
    capit = []
    capitt =[]
    for a in corpus:
        count = 0 
        b = a.split()
        for c in b:
            if c.isupper():
                count = count + 1;
        capit.append(False if count == 0 else True)      
        capitn.append(count);
        t =[]
        t.append(False if count == 0 else True)    
        t.append(count)
        capitt.append(t)
        
    #with open('capitalisation.txt', 'w') as data_in:
        #for i in range(0,len(corpus)):
            #data_in.write("%f\t%f\n" %(capitn[i],capit[i]))
    
    return capitn,capit,capitt

if __name__ == "__main__":
    # Experiment settings

    # Dataset: SemEval2018-T4-train-taskA.txt or SemEval2018-T4-train-taskB.txt
    DATASET_FP = "./SemEval2018-T3-train-taskA.txt"
    TASK = "A" # Define, A or B
    FNAME = './predictions-task' + TASK + '.txt'
    PREDICTIONSFILE = open(FNAME, "w")
      

    K_FOLDS = 10 # 10-fold crossvalidation
    CLF = LinearSVC() # the default, non-parameter optimized linear-kernel SVM

    # Loading dataset and featurised simple Tfidf-BoW model
    corpus, y = parse_dataset(DATASET_FP)
    
    X = featurize(corpus)
    punc,ellips,punclist = punctuation(corpus)
    capitn,capit,capitt = capitalisation(corpus)
    m = X                              #0.632568426873
    #m = np.hstack((X,capitt))         #0.632389331867
    #m = np.hstack((m,punclist))       #0.634266886326
    #m = np.hstack((capitt,punclist))  #0.570984225728
    #combine all three : 0.634816035146
    class_counts = np.asarray(np.unique(y, return_counts=True)).T.tolist()
    print (class_counts)
    
    # Returns an array of the same size as 'y' where each entry is a prediction obtained by cross validated
    predicted = cross_val_predict(CLF, m, y, cv=K_FOLDS)
    #libsvm
    #predicted = libsvm.cross_validation(X.toarray(), np.asarray(y,'float64'), 5, kernel = 'rbf')

    # Modify F1-score calculation depending on the task
    if TASK.lower() == 'a':
        score = metrics.f1_score(y, predicted, pos_label=1)
    elif TASK.lower() == 'b':
        score = metrics.f1_score(y, predicted, average="macro")
    print ("F1-score Task", TASK, score)
    for p in predicted:
        PREDICTIONSFILE.write("{}\n".format(p))
    PREDICTIONSFILE.close()
    
    
    
