#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 16:17:25 2023

@author: be
"""

import os
cd = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(1, cd+'/LanguageData')

import tensorflow as tf
import pandas as pd
import numpy as np
from rich.progress import track

#Local Files
import preprocessData
import WordLanguageModel as wlm
from LanguageDecoder import LanguageDe

def main():
    data = preprocessData.getWordData().getWords()

    LANGUAGES = pd.read_csv(cd + '/LanguageData/languages.csv').dropna().code
    LANG_NUMBER = LANGUAGES.size
    
    wlmHandler = wlm.WordLanguageModel(languages = LANGUAGES)
    model = wlmHandler.getModel()
    vectorize_layer = wlmHandler.getVectorizeLayer()
         
    def encodeLanguage(string):
        n = np.zeros(LANG_NUMBER).astype('float32')
        for i in range(LANG_NUMBER):
            if string == LANGUAGES[i]:
                n[i] = 1
                break
        return tf.convert_to_tensor(n.astype(np.float32))
    
    def decodeLanguage(b):
        assert type(b) == np.ndarray, "The input has to be a numpy.ndarray"
        assert b.size == LANGUAGES.shape[0], "The input doesnt have the correct size"
        a = b[0]
        s = 0
        index = 0
        for i in range(len(a)):
            if (a[i]>s):
                s = a[i]
                index  = i
                
        return LANGUAGES[index]
    
    # In[]
    
    print("Vectorising Tokens")
    vectorize_layer.adapt(tf.constant(data.drop(['language'], axis=1)))
    
    X_train = data['word']
    Y_train = data['language']
    
    print("Encoding Y Values")
    Y_array = np.zeros([data.count()[0],LANG_NUMBER])
    
    for i in track(range(data.count()[0])):
        Y_array[i] = encodeLanguage(Y_train.iat[i])
        
    
    reshaped_arr = X_train.values.reshape((X_train.size, 1))
    
    
    # In[]
    
    print ("Training the model")
    model.fit(reshaped_arr, Y_array, epochs = 4, validation_split=0.2)
    
    
    
    # In[]
    
    try:
    
        model.save(cd +'/model/')
    except:
        print("Data could not be stored properly")
    
    ld = LanguageDe()
    print("Please input a word, I will guess the language! Enter 'quit' to stop the program.")
    while True:
        z = input()
        if z == "quit" or z == "stop" or z == "quit()":
            exit()
        print(ld.decode(model.predict([z])))
    

if __name__=="__main__":
    main()
