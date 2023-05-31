#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:08:03 2023

@author: be
"""
import time
import pandas as pd
from util import *
import tensorflow as tf
from tensorflow import keras

def loadModelInfo():
    data = {'coin':[], 'filename':[],'date':[], 'range':[]}
    availableModels = pd.DataFrame(data)
    availableModels.set_index("coin", inplace = True)
    try:
        availableModels = pd.read_pickle("modelInfo.txt")
    except Exception as error:
        storeModelInfo(availableModels)
        print(error)
        
    return availableModels

def storeModelInfo(m):
    m.to_pickle("modelInfo.txt")
    
def getModels():
    availableModels = loadModelInfo()
    d = dict()
     
    for i in range(availableModels.shape[0]):
        print("Load: ", availableModels.index[i])
        d2 = dict()
        d2["symbol"] = availableModels.index[i]
        d2["range"]  = availableModels.iat[i,2]
        d2["model"]  = tf.keras.models.load_model(availableModels.iat[i,0])
        d[availableModels.index[i]] = d2
    return d

def saveModels(coin, model, r):
    path = coin + ".h5"
    tf.keras.models.save_model(model, path)
    a = loadModelInfo()
    a.loc[coin] = path
    row = a.loc[coin]
    row['filename'] = path
    row['date'] = time.time()
    row['range'] = r
    print(row)
    a.loc[coin] = row
    storeModelInfo(a.copy())
    
    