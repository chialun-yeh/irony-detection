#!/usr/bin/env python3

'''
example.py

Benchmark system for the SemEval-2018 Task 3 on Irony detection in English tweets.
The system makes use of token unigrams as features and outputs cross-validated F1-score.

Date: 1/09/2017
Copyright (c) Gilles Jacobs & Cynthia Van Hee, LT3. All rights reserved.
'''

from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import metrics
import numpy as np
import logging
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn


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
    with open(fp, 'rt') as data_in:
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
    Tokenizes and creates TF-IDF BoW vectors.
    :param corpus: A list of strings each string representing document.
    :return: X: A sparse csr matrix of TFIDF-weigted ngram counts.
    '''

    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words="english")
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()
    # print(vectorizer.get_feature_names()) # to manually check if the tokens are reasonable
    return X

def senti_featurize(corpus):
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    lemma = WordNetLemmatizer()
    X  = [];
    for line in corpus:
        token = tokenizer.tokenize(line)
        token = [word for word in token if word not in stopwords.words('english')]
        token = [lemma.lemmatize(word) for word in token]
        poseachtweet=[]
        negeachtweet=[]
        for lem in token:
            a,b=0,0
            syn = list(swn.senti_synsets(lem))
    
            for sy in syn:
                a+=sy.pos_score()
                b+=sy.neg_score()
    
            if(len(syn)!=0):
                a = a/len(syn)
                b = b/len(syn)
    
    
            poseachtweet.append(a)
            negeachtweet.append(b)
        if(len(token)!=0):
            max_pos = max(poseachtweet)
            max_neg = max(negeachtweet)
            pos_sum = sum(poseachtweet)
            neg_sum = sum(negeachtweet)
            imbal = max(poseachtweet) - min(negeachtweet)
            senti_avg = (pos_sum-neg_sum)/len(token)
            positive_gap = max_pos - senti_avg
            negative_gap = max_neg - senti_avg
    
    
        else:
            max_pos = 0
            max_neg =0
            pos_sum = 0
            neg_sum =0
            senti_avg =0
            imbal = 0
            positive_gap = 0
            negative_gap = 0
            
    
        #positive_sum.append(pos_sum)
        #negative_sum.append(neg_sum)
        #averagesenti.append(senti_avg)
        #imbalance.append(imbal)
        #posgap.append(positive_gap)
        #neggap.append(negative_gap)
        X.append([float(pos_sum), float(neg_sum), float(imbal), float(senti_avg), float(positive_gap), float(negative_gap)])
    return X


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
    Y = senti_featurize(corpus)
    
    Z = np.hstack((X,Y))

    class_counts = np.asarray(np.unique(y, return_counts=True)).T.tolist()
    print (class_counts)
    
    # Returns an array of the same size as 'y' where each entry is a prediction obtained by cross validated
    predicted = cross_val_predict(CLF, Z, y, cv=K_FOLDS)
    
    # Modify F1-score calculation depending on the task
    if TASK.lower() == 'a':
        score = metrics.f1_score(y, predicted, pos_label=1)
    elif TASK.lower() == 'b':
        score = metrics.f1_score(y, predicted, average="macro")
    print ("F1-score Task", TASK, score)
    for p in predicted:
        PREDICTIONSFILE.write("{}\n".format(p))
    PREDICTIONSFILE.close()
    
    
    
