#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 17:52:40 2018

@author: xuecho
"""
import pandas as pd
import numpy as np
import pickle

def save_obj(obj, name):
    f = open(name + ".dat", "wb")
    pickle.dump(obj, f)
    f.close()
    
def load_obj(name):
    return pickle.load(open(name + ".dat", "rb"))

# reading the file
df = pd.read_csv('Emoji_Sentiment_Data_v1.0.csv')
data = df[['Emoji','Unicode codepoint','Negative','Neutral','Positive']]

sLength = len(data['Emoji'])
data.loc[:,'sentiment'] = pd.Series(0, index=data.index)
data['sentiment'] = data['Positive']/(data['Negative']+data['Neutral']+data['Positive'])
data.loc[:,'polarity'] = pd.Series(0, index=data.index)
data['polarity'] = data['sentiment'] > 0.5

save_obj(data, 'emoji_sentiment_list')
#a = load_obj('emoji_sentiment_list')
