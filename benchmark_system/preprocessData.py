#!/usr/bin/env python3

import re
import os
import string
from nltk import pos_tag
from nltk.corpus import wordnet
import re
from collections import Counter
import emoji
from emoji.unicode_codes import UNICODE_EMOJI
import unicodedata
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
import nltk
from nltk.tokenize import TweetTokenizer
import numpy as np

def parse_dataset(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    #standard preprocessing: remove url and replace @tag with user
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
                tweet = re.sub('@[^\s]+','user', tweet)
                if(len(tweet) != 0):
                    y.append(label)
                    corpus.append(tweet)
    return corpus, y

if __name__ == "__main__":
    # Experiment settings

    trn_dataset = "../datasets/train/SemEval2018-T3-train-taskA_emoji.txt"
    fname = './processed-trn-data.txt'
    # Load dataset
    corpus, y = parse_dataset(trn_dataset)
    print(len(corpus))

    feat = extract_entities(corpus[1:50])
    
    #save processed corpus
    with open (fname,'w', encoding='utf-8') as datafile:
        for i in range(len(corpus)):
            datafile.write(str(y[i]) + " " + corpus[i] + "\n" )


