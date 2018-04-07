#!/usr/bin/env python3

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
                pattern = 'http.+(\s)?'
                tweet = re.sub(pattern, '', tweet, flags=re.MULTILINE) #remove url
                tweet = re.sub('@[^\s]+','user', tweet) #replace @tag with user
                if(len(tweet) != 0):
                    y.append(label)
                    corpus.append(tweet)
    return corpus, y

def parse_testset(fp):
    corpus = []
    with open(fp, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                pattern = '(http.+(\s)?)|((@\w*\d*(\s)?))'
                tweet = re.sub(pattern, '', tweet, flags=re.MULTILINE) #remove url
                corpus.append(tweet)
    return corpus

def get_label(fp):
    label = []
    with open (fp, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                lab = int(line.split("\t")[1])
                label.append(lab)
    return label

english_stemmer = SnowballStemmer('english')

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

#### Lexical features #######
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

def punctuation(corpus):
    punc = []
    ellips = []
    punclist =[]
    pat2 = '[(\.\.)]+'
    
    for a in corpus:
        punc.append(a.count('!')+a.count('?'))
        ellips.append(len(re.findall(pat2,a)))
        t =[]
        t.append(a.count('!')*10)
        t.append(a.count('?')*10)
        t.append(len(re.findall(pat2,a))*10)
        punclist.append(t)
    return punc,ellips,punclist

def capitalisation(corpus):
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=True)
    capitn = [] #number of words of all capitalized letter
    capit = []  #True: contain words of letter all capitalized
    capitl = [] #number of words with capitalized letter
    capitt =[]  #list of three features
    for a in corpus:
        count = 0 
        count_l = 0 
        b = tokenizer.tokenize(a)
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
    return capitn,capit,capitl,capitt

def sentenceLength(corpus):
    '''
    length: #word in the tweet
    character: mean length of each word in the tweet
    '''
    tokenizer = TweetTokenizer(preserve_case=False)
    length = []
    character = []
    for a in corpus:
        tokens = tokenizer.tokenize(a)
        length.append([len(tokens)])
        cnt = 0
        for t in tokens:
            cnt += len(t)
        character.append([float(cnt)/float(len(tokens))])
    return length, character

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

def laughing(corpus):
    '''
    number of occurance of hahaha, lol, rofl, and lmao
    '''
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

#### Syntactic Features###########
def pos_features(corpus):
    tknzr = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = [tknzr.tokenize(c) for c in corpus]
    tagged_tokens = [pos_tag(sentence) for sentence in tokens]
    tags = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','MD','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ','RB','RBR','RBS','RP','WP','WDT','WRB','PDT','PRP','PRP$', 'UH','SYM','TO']
    feat1=[] #feat1: for each tags, whether it occurs or not
    feat2=[] #feat2: whether it occurs 0, 1 or >2 times
    feat3=[] #feat3: the frequency of tags as number
    feat4=[] #feat4: the frequency of tags as percentage 
    feat5=[] #feat6: whether there is a clash between verb tense
    counts = dict()
    feat=[]
    for t in tags:
        counts[t] = 0
    for i in range(len(tagged_tokens)):
        feat1.append([])
        feat2.append([])
        feat3.append([])
        feat4.append([])
        for w in tagged_tokens[i]:  
            if w[1] in counts:
                counts[w[1]] += 1  

        for key, value in counts.items():
            if len(tagged_tokens) != 0:
                feat3[i].append(value)
                feat4[i].append(float(value)/float(len(tagged_tokens)))
                feat1[i].append(0 if value==0 else 1)
                if value ==0:
                    feat2[i].append(0)
                elif value == 1:
                    feat2[i].append(1)
                else:
                    feat2[i].append(2)
            else:
                feat1[i].append(0)
                feat2[i].append(0)
                feat3[i].append(0)
                feat4[i].append(0)
            counts[key] = 0

        if (counts['VBP'] >0 or counts['VBZ']>0) and counts['VBD']>0:
            feat5.append([1])
        else:
            feat5.append([0])
    feat = np.concatenate([feat1,feat2,feat3,feat4,feat5],axis=1)
    return feat

def extract_entities(corpus):
    ne_list= ['LOCATION', 'ORGANIZATION', 'PERSON', 'DURATION','DATE', 'CARDINAL', 'PERCENT', 'MONEY', 'MEASURE', 'FACILITY', 'GPE']
    feat1=[] #feat1: if there is NE
    feat2=[] #feat2: #of different NEs
    feat3=[] #feat3: #of NEs
    entities=[]
    for a in corpus:
        e_n =[]
        e_l =[]
        parse_tree = ne_chunk(pos_tag(a.split()), binary=False)
        count = 0;
        for t in parse_tree.subtrees():
            if t.label() in ne_list:
                e_n.append(t)
                e_l.append(t.label())
                count = count + len(t);

        if count > 0:
            feat1.append([1])
        else:
            feat1.append([0])
        feat2.append([len(set(e_l))])
        feat3.append([count])
        entities.append(e_n)
    feat = np.concatenate([feat1, feat2, feat3], axis=1)
    return entities, feat

#### Sentiment Lexicon Features ###############
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


def featurize(corpus):
    X = tfidf_vectors(corpus)
    X1 = senti_features(corpus)
    X2 = char_flooding(corpus)
    X3 = pos_features(corpus) #0.6514 
    punc,ellips,X4 = punctuation(corpus)
    capitn,capit,capitl,X5 = capitalisation(corpus)
    X6 = laughing(corpus)
    X7 = quotes(corpus)
    X8 = synonym(corpus)
    sentLen, wordLen = sentenceLength(corpus)
    entities,X10 = extract_entities(corpus)
    X11,emojlist,X12 = emojiList(corpus)
    X13,X14,emolist = emoticonList(corpus)

    Z = np.hstack((X,X1,X2,X3,X4,X5,X6,X7,X8,sentLen,wordLen,X10,X11,X12,X13,X14))

    return Z


if __name__ == "__main__":
    # Experiment settings

    trn_dataset = "../datasets/train/SemEval2018-T3-train-taskA_emoji.txt"
    FNAME = './predictions.txt'
    PREDICTIONSFILE = open(FNAME, "w")

    K_FOLDS = 10 # 10-fold crossvalidation
    CLF = LinearSVC() # the default, non-parameter optimized linear-kernel SVM
    #CLF = DecisionTreeClassifier(random_state=0)
    #CLF = GaussianNB()
    #CLF = LogisticRegression()

    # Load dataset
    corpus, y = parse_dataset(trn_dataset) #3802 in total
    Xtrn = featurize(corpus)
    print(np.asarray(Xtrn).shape)
    

    class_counts = np.asarray(np.unique(y, return_counts=True)).T.tolist()
    print (class_counts)
    
    # Returns an array of the same size as 'y' where each entry is a prediction obtained by cross validated
    predicted = cross_val_predict(CLF, Xtrn, y, cv=K_FOLDS)
    #predicted = libsvm.cross_validation(Z, np.asarray(y,'float64'), 5, kernel = 'rbf')
    

    #Testing
    #tst_dataset = "../datasets/test/tweet/SemEval2018-T3_input_test_taskA.txt"
    #test_corpus = parse_testset(tst_dataset)
    #tst = featurize(test_corpus)
    # Initialize classifier
    # clsf = LinearSVC()
    # Train classifier
    #model = clsf.fit(Xtrn, y)
    #predict
    #pred = clsf.predict(Xtst)


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
