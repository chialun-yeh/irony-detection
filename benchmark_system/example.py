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
from sklearn.svm import libsvm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np
import logging
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag
import re
from collections import Counter


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

english_stemmer = SnowballStemmer('english')

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


def featurize(corpus):
    '''
    Tokenizes and creates TF-IDF BoW vectors.
    :param corpus: A list of strings each string representing document.
    :return: X: A sparse csr matrix of TFIDF-weigted ngram counts.
    '''
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    #vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words="english")
    vectorizer = StemmedTfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words="english", ngram_range=(1,2))
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()
    return X

def char_flooding(corpus):
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=True)
    X = [];
    for line in corpus:
        token = tokenizer.tokenize(line)
        token = [word for word in token]
        count=1
        for i in range(1,len(word)):
            if word[i-1]==word[i]:
                count+=1
                if count >= 3:
                    X.append([1])
                    break;
            else :
                count=1
        if count < 3:
            X.append([0])
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

def pos_feat(corpus, stop_words=True, strip_url=True):
    '''
    Tokenizes and creates TF-IDF BoW vectors.
    :param corpus: A list of strings each string representing document.
    :return: X: A sparse csr matrix of TFIDF-weigted ngram counts.
    '''
    #[re.sub(r'.+', ' ', c) for c in corpus]
    print(corpus[1])
    tknzr = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = [tknzr.tokenize(c) for c in corpus]
    

    if stop_words:
        filterd = []
        stopWords = set(stopwords.words('english'))
        for i in range(len(tokens)):
            filterd.append([])            
            for word in tokens[i]:
                #if word not in stopWords and word != '.' and word != ',' and word != '...':
                if word != '.' and word != ',' and word != '...':
                    filterd[i].append(word)
            filterd[i] = pos_tag(filterd[i])

    tags = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','MD','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ','RB','RBR','RBS','RP','WP','WDT','WRB','PDT','PRP','PRP$', 'UH','SYM','TO']
    freq = []
    freq3Level = []
    counts = dict()
    for t in tags:
        counts[t] = 0
    for i in range(len(filterd)):
        freq.append([])
        freq3Level.append([])
        for w in filterd[i]:  
            if w[1] in counts:
                counts[w[1]] += 1        
        for key, value in counts.items():
            freq[i].append(value)
            if value ==0:
                freq3Level[i].append(0)
            elif value == 1:
                freq3Level[i].append(1)
            else:
                freq3Level[i].append(2)
            counts[key] = 0
    #percent = np.divide(freq, float(np.sum(np.asarray(freq), axis=1).T))
    X = np.concatenate((freq,freq3Level), axis=1)
    #X = np.concatenate((X,percent), axis=1)
    return X
 
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
        punc.append(a.count('!')+a.count('?'))
        ellips.append(len(re.findall(pat2,a)))
        t =[]
        t.append(a.count('!'))
        t.append(a.count('?'))
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
        capit.append(1 if count == 0 else 0)      
        capitn.append(count);
        t =[]
        t.append(1 if count == 0 else 0)    
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
    #CLF = DecisionTreeClassifier(random_state=0)
    #CLF = GaussianNB()

    # Loading dataset and featurised simple Tfidf-BoW model
    corpus, y = parse_dataset(DATASET_FP)
    X = featurize(corpus)
    X1 = senti_featurize(corpus)
    X2 = char_flooding(corpus)
    X3 = pos_feat(corpus)
    punc,ellips,X4 = punctuation(corpus)
    capitn,capit,X5 = capitalisation(corpus)
    
    Z = np.hstack((X,X1,X2,X3,X4,X5))

    class_counts = np.asarray(np.unique(y, return_counts=True)).T.tolist()
    print (class_counts)
    
    # Returns an array of the same size as 'y' where each entry is a prediction obtained by cross validated
    predicted = cross_val_predict(CLF, Z, y, cv=K_FOLDS)
    #predicted = libsvm.cross_validation(Z, np.asarray(y,'float64'), 5, kernel = 'rbf')
    
    # Modify F1-score calculation depending on the task
    if TASK.lower() == 'a':
        score = metrics.f1_score(y, predicted, pos_label=1)
    elif TASK.lower() == 'b':
        score = metrics.f1_score(y, predicted, average="macro")
    print ("F1-score Task", TASK, score)
    for p in predicted:
        PREDICTIONSFILE.write("{}\n".format(p))
    PREDICTIONSFILE.close()
    
    
    
