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
from nltk.corpus import wordnet
import re
from collections import Counter
import emoji
from emoji.unicode_codes import UNICODE_EMOJI
import unicodedata
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.tokenize.casual import EMOTICON_RE
from EmoticonDetector import *
from EmojiDetector import *


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
    with open(fp, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                label = int(line.split("\t")[1])
                tweet = line.split("\t")[2]
                pattern = '(http.+(\s)?)|((@\w*\d*(\s)?))'
                tweet = re.sub(pattern, '', tweet, flags=re.MULTILINE) #remove url
                y.append(label)
                corpus.append(tweet)

    return corpus, y

english_stemmer = SnowballStemmer('english')

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


def tfidf_vectors(corpus):
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
        flood = 0
        for word in token:
            for i in range(1,len(word)):
                if word[i-1]==word[i]:
                    count+=1
                    if count >= 3:
                        flood+=1
                else :
                    count=1
                
        X.append([flood]*10)
    # print(vectorizer.get_feature_names()) # to manually check if the tokens are reasonable
    return X

def senti_features(corpus):
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
            
        X.append([float(pos_sum), float(neg_sum), float(imbal), float(senti_avg), float(positive_gap), float(negative_gap)])
    return X

def pos_features(corpus):
    tknzr = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = [tknzr.tokenize(c) for c in corpus]
    tagged_tokens = [pos_tag(sentence) for sentence in tokens]

    tags = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','MD','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ','RB','RBR','RBS','RP','WP','WDT','WRB','PDT','PRP','PRP$', 'UH','SYM','TO']
    freq = []
    freq3Level = []
    counts = dict()
    for t in tags:
        counts[t] = 0
    for i in range(len(tagged_tokens)):
        freq.append([])
        freq3Level.append([])
        for w in tagged_tokens[i]:  
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
    X = np.concatenate((freq,freq3Level), axis=1)
    return X

def laughing(corpus):
    #pattern = '(http.+(\s)?)|((@\w*\d*(\s)?))'
    #corp1 = []

    #for str in corpus:
    #    str1 = re.sub(pattern, '', str, flags=re.MULTILINE)
    #    corp1.append(str1)


    pat = '([aA]*[hH]+[Aa]+[Hh][HhAa]*|[Oo]?[Ll]+[Oo]+[Ll]+[OolL]*|[Rr][oO]+[Ff]+[lL]+|[Ll]+[Mm]+[Aa]+[oO]+).'
    laughing = []

    for a in corpus:
        b= re.findall(pat,a)
        laughing.append([len(b)]*10)
    
    return laughing

def quotes(corpus):
  
    pat = '["]+'
    quotes = []

    for a in corpus:
        b= re.findall(pat,a)
        quotes.append([len(b)]*10)
    
    return quotes
 
def punctuation(corpus):
    punc = []
    ellips = []
    punclist =[]
    pat2 = '[(\.\.)]+'
    
    #pattern = '(http.+(\s)?)|((@\w*\d*(\s)?))'
    #corp1 = []
    #for str in corpus:
    #    str1 = re.sub(pattern, '', str, flags=re.MULTILINE)
    #    corp1.append(str1)
    
    for a in corpus:
        punc.append(a.count('!')+a.count('?'))
        ellips.append(len(re.findall(pat2,a)))
        t =[]
        t.append(a.count('!')*10)
        t.append(a.count('?')*10)
        t.append(len(re.findall(pat2,a))*10)
        punclist.append(t)
        
    #with open('punctuation.txt', 'w') as data_in:
        #for i in range(0,len(corpus)):
            #data_in.write("%f\t%f\n" %(punc[i],ellips[i]))
            
    return punc,ellips,punclist

def capitalisation(corpus):
    capitn = [] #number of words of all capitalized letter
    capit = []  #True: contain words of letter all capitalized
    capitl = [] #number of words with capitalized letter
    capitt =[]  #list of three features
    for a in corpus:
        count = 0 
        count_l = 0 
        b = a.split()
        for c in b:     # c: word
            if c.isupper():
                count = count + 1;
            if any(d.isupper() for d in c):
                count_l = count_l + 1;
        capit.append(False if count == 0 else True)      
        capitn.append(count);
        capitl.append(count_l);
        t =[]
        t.append(False if count == 0 else True)    
        t.append(count)
        t.append(count_l)
        capitt.append(t)
    #with open('capitalisation.txt', 'w') as data_in:
        #for i in range(0,len(corpus)):
            #data_in.write("%f\t%f\n" %(capitn[i],capit[i]))
    
    return capitn,capit,capitl,capitt

def sentenceLength(corpus):
    length = []
    for a in corpus:
        b = a.split()
        t=[]
        t.append(len(b))
        length.append(t)
    return length

def extract_entities(corpus):
    entities = []
    entitiesCount = []
    for a in corpus:
        e_n =[]
        e_c =[]
        parse_tree = ne_chunk(pos_tag(a.split()), binary=True)
        count = 0;
        for t in parse_tree.subtrees():
            if t.label() == 'NE':
                e_n.append(t)
                count = count + 1;
        e_c.append(count);
        entitiesCount.append(e_c);
        entities.append(e_n);
    return entities,entitiesCount

def emojiList(corpus):
    corpusNoEmo =[] # to store emoji name appended at the end of text
    emolist = []
    emocount =[]
    emoTorF =[]
    for a in corpus:
        b = a.split()
        count = 0
        t = [] # store emoji string
        ct = [] # store count
        TorF =[] # possive or negative
        all_emoji = []
        
        for char in b:
            s = ""
            if char in emoji.UNICODE_EMOJI:
                all_emoji.append(char)
                count = count + 1;
                #convert emoji into name and append at the end of text
                #a = a + unicodedata.name(char) + " " 
                s+= str(unicodedata.name(char)) + "," 
            
        if(len(all_emoji) != 0):
            ed = EmojiDetector()
            all_emoji.sort();
            tf = (-1, 1)[ed.is_positive(all_emoji[0])]
            TorF.append(tf)
        else:
            TorF.append(0)
                
        t.append(s)
        ct.append(count)
        emolist.append(t)
        emocount.append(ct)
        emoTorF.append(TorF)
 
    return emocount,emolist,emoTorF

def emoticonList(corpus):
    emolist = []
    emocount =[]
    emoTorF =[]
    for a in corpus:
        ct = [] # store count
        TorF =[] # possive or negative
        all_emoticons = EMOTICON_RE.findall(a)
        ct.append(len(all_emoticons))
        if(len(all_emoticons) != 0):
            ed = EmoticonDetector()
            all_emoticons.sort();
            tf = (-1, 1)[ed.is_positive(all_emoticons[0])]
            TorF.append(tf)
        else:
            TorF.append(0)
        emolist.append(all_emoticons)
        emocount.append(ct)
        emoTorF.append(TorF)   
    return emocount,emoTorF, emolist

def preprocessing(corpus):
    corpusNoEmo =[]
    emoji_pattern = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    for a in corpus:
        #remove emojis from text corpus
        a  = re.sub(emoji_pattern, '', a)
        #remove any url to URL
        a = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',a)
        #Convert any @Username to "AT_USER"
        a = re.sub('@[^\s]+','AT_USER',a)
        #Remove additional white spaces
        a = re.sub('[\s]+', ' ', a)
        a = re.sub('[\n]+', ' ', a)
        #Remove not alphanumeric symbols white spaces
        a = re.sub(r'[^\w]', ' ', a)
        #Replace #word with word
        a = re.sub(r'#([^\s]+)', r'\1', a)
        corpusNoEmo.append(a)         

    return corpusNoEmo


def synonym (corpus):
    tknzr = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    synonyms = []
    for line in corpus:
        cnt = 0;
        tokens = tknzr.tokenize(line)
        token = [word for word in tokens if word not in stopwords.words('english')]
        for word in token:
            wordsyn = [] 
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    wordsyn.append(l.name())
            cnt += len(set(wordsyn))
        if(len(token)==0):
            synonyms.append([0])
        else:
            synonyms.append([cnt/len(token)]) 
    return synonyms


if __name__ == "__main__":
    # Experiment settings

    trn_dataset = "../datasets/train/SemEval2018-T3-train-taskA_emoji.txt"
    FNAME = './predictions-task.txt'
    PREDICTIONSFILE = open(FNAME, "w")

    K_FOLDS = 10 # 10-fold crossvalidation
    CLF = LinearSVC() # the default, non-parameter optimized linear-kernel SVM
    #CLF = DecisionTreeClassifier(random_state=0)
    #CLF = GaussianNB()
    #CLF = LogisticRegression()

    # Load dataset
    corpus, y = parse_dataset(trn_dataset)
    

    X = tfidf_vectors(corpus)
    X1 = senti_features(corpus)
    X2 = char_flooding(corpus)
    X3 = pos_features(corpus)
    punc,ellips,X4 = punctuation(corpus)
    capitn,capit,capitl,X5 = capitalisation(corpus)
    X6 = laughing(corpus)
    X7 = quotes(corpus)
    X8 = synonym(corpus)
    X9 = sentenceLength(corpus)
    entities,X10 = extract_entities(corpus)
    X11,emojlist,X12 = emojiList(corpus)
    X13,X14,emolist = emoticonList(corpus)
                           #0.632568426873

    Z = np.hstack((X,X12))
    #Z = np.hstack((X,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14))
    class_counts = np.asarray(np.unique(y, return_counts=True)).T.tolist()
    #print (class_counts)
    
    # Returns an array of the same size as 'y' where each entry is a prediction obtained by cross validated
    predicted = cross_val_predict(CLF, Z, y, cv=K_FOLDS)
    #predicted = libsvm.cross_validation(Z, np.asarray(y,'float64'), 5, kernel = 'rbf')
    

    score = metrics.f1_score(y, predicted, pos_label=1)
    acc = metrics.accuracy_score(y, predicted)
    preci = metrics.precision_score(y, predicted)
    recall = metrics.recall_score(y, predicted)

    print ("F1-score:", score)
    print ("Accuracy:", acc)
    print ("Precision:", preci)
    print ("Recall:", recall)

    for p in predicted:
        PREDICTIONSFILE.write("{}\n".format(p))
    PREDICTIONSFILE.close()
