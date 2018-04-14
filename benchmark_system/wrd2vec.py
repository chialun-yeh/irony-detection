# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 04:02:35 2018

@author: Pooja Prajod
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import re
from sklearn import metrics
from sklearn.svm import SVC
from gensim.scripts.glove2word2vec import glove2word2vec


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
    with open(fp, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                label = int(line.split("\t")[1])
                tweet = line.split("\t")[2]
                pattern = 'http.+(\s)?'
                tweet = re.sub(pattern, '', tweet, flags=re.MULTILINE) #remove url
                tweet = re.sub('@[^\s]+','user', tweet) #replace @tag with user
                if(len(tweet) != 0):
                    y.append(label)
                    corpus.append(tweet)
    return corpus, y

#loading the model
#model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
def loadGloveModel(gloveFile):
    f = open(gloveFile,'r', encoding='utf-8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

#model = loadGloveModel('glove.twitter.27B.200d.txt')
model = gensim.models.KeyedVectors.load_word2vec_format('word2vec_twitter_model.bin', binary=True, unicode_errors='ignore')
#start testing
trn_dataset = "../datasets/train/SemEval2018-T3-train-taskA_emoji.txt"

# Load dataset
corpus, y = parse_dataset(trn_dataset)

tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)


lemma = WordNetLemmatizer()
porter = nltk.PorterStemmer()
X  = []
dim = len(model['eat'])
i=0
for line in corpus:
    token = tokenizer.tokenize(line)
    token = [word for word in token if word not in stopwords.words('english')]
    #token = [porter.stem(i.lower()) for i in token] 
    #token = [lemma.lemmatize(word) for word in token]
    X.append(np.mean([model[w] for w in token if w in model] or [np.zeros(dim)], axis=0))
    i = i + 1

predicted = cross_val_predict(SVC(C=20), X, y, cv=10)  
score = metrics.f1_score(y, predicted, pos_label=1)
acc = metrics.accuracy_score(y, predicted)
preci = metrics.precision_score(y, predicted)
recall = metrics.recall_score(y, predicted)

print ("F1-score:", score)
print ("Accuracy:", acc)
print ("Precision:", preci)
print ("Recall:", recall)  


