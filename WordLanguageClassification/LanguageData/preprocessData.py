#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:08:07 2023

@author: be
"""
import sys
import os
cd = os.path.dirname(os.path.realpath(__file__))

import pandas as pd
    
class getWordData():
    def containsNumber(value):
        return any([char.isdigit() for char in value])

    def __init__(self, wordPerLanguage = -1):  
        languages = pd.read_csv(cd + '/languages.csv').dropna()
        languageArray = []
        languageIdentifierArray = []
        
        wordCount = 10e20
        
        for i in languages['code']:
            data = pd.read_csv(cd + "/" + str(i) + '.csv', encoding = languages.loc[languages['code'] == i,'encoding'].item()).sample(frac=1)
            data = data.dropna()
            data.rename(columns ={data.columns[0]:"word"}, inplace = True)
            data = data[data['word'].str.len()>3]
            if i != "de_DE":
                data = data[data['word'].str.lower() == data['word']]
                
            wordsN = data.count()[0]
            wordCount = min(wordCount,wordsN)
                
            label = [i for _ in range(wordsN)]
            data['language'] = label
            languageArray.append(data)
            languageIdentifierArray.append(i)
            print("Data File read: " + str(i))
            
        totalWordsLocal = pd.DataFrame()
        if wordPerLanguage != -1 :
            wordCount = min(wordCount, wordPerLanguage)
            
        for u in range(len(languageArray)):
            totalWordsLocal = pd.concat([totalWordsLocal,languageArray[u].head(wordCount)], ignore_index=True)
        
        self.totalWords = totalWordsLocal
        
    def getWords(self):
        return self.totalWords.sample(frac=1)
    
if __name__=="__main__":
    print("Pls dont run the module like this")
    sys.exit()