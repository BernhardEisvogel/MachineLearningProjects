#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:34:51 2023

@author: be
"""
import os
cd = os.path.dirname(os.path.realpath(__file__))

import tensorflow as tf
import pandas as pd
import numpy as np

class LanguageDe:
    def __init__(self):  
        self.LANGUAGES = pd.read_csv(cd + '/LanguageData/languages.csv').dropna().language
        
    def decode(self,b):
        assert type(b) == np.ndarray, "The input has to be a numpy.ndarray"
        assert b.size == self.LANGUAGES.shape[0], "The input doesnt have the correct size"
        a = b[0]
        s = 0
        index = 0
        for i in range(len(a)):
            if (a[i]>s):
                s = a[i]
                index  = i
                
        return self.LANGUAGES[index]