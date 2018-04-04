#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:41:28 2018

@author: xuecho
"""

class EmojiDetector:
    emojis = {}
    
    def __init__(self, emoji_file="emoji_sentiment_list"):
        import numpy as np
        data = self.load_obj(emoji_file)
        lst = np.array(data).tolist()
        positive = True
        for line in lst:
            positive = line[6];
            self.emojis[line[0]] = positive

    def is_positive(self, emoji):
        if emoji in self.emojis:
            return self.emojis[emoji]
        return False

    def load_obj(self,name):
        import pickle
        return pickle.load(open(name + ".dat", "rb"))
