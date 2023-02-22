#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 22:28:55 2023

@author: be
"""
import sys
import os
cd = os.path.dirname(os.path.realpath(__file__))
import tensorflow as tf
from tensorflow.keras import layers

class WordLanguageModel:
    vectorize_layer = tf.keras.layers.TextVectorization()
    model =  tf.keras.Model()
    def predict(self, string):
        assert type(string) == str, "The input has to be a string"
        return self.decodeLanguage(self.model.predict([string]))
    
    def getModel(self):
        return self.model
    
    def getCompleteModel(self):
        decodeLayer = layers.Lambda(self.decodeLanguage)(self.model.outputs)
        newModel = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=decodeLayer)
        return newModel

    def getVectorizeLayer(self):
        return self.vectorize_layer
    
    def __init__(self, languages, read = False):
        assert type(read) == bool, "The 'read' value has to be true or false"
        
        self.languages = languages
        
        def encodeWord(string):
            return tf.strings.lower(string)
    
        INPUT_SHAPE     = max(17,int(self.languages.count() * 1.6))
        MAX_TOKENS      = 2400
        MIDDLE_LAYER    = int(INPUT_SHAPE/1.2)
        
        self.vectorize_layer = layers.TextVectorization(
            standardize="lower",
            max_tokens=MAX_TOKENS,
            output_mode='int',
            split="character",
            ngrams=3,
            output_sequence_length=INPUT_SHAPE
            )
        
        '''
        I had the idea to improve the model by introucing the UTF-8  mean value
        as an input so that the model can better differentiatie between different keyboards,
        but it doesnt make the model better in any way.
        
        def getAsciiMean(s):
            return tf.reshape(tf.math.reduce_mean(tf.strings.unicode_decode(s,'UTF-8')/100000, 2), (-1,1))
        asciimean=layers.Lambda(getAsciiMean)(visible)
        asciimean._name = "Layer that calculates ASCII-Mean values"
        normalize = tf.keras.layers.Normalization()(asciimean)
        normalize._name = "Normalize the Asciimean input"
        
        concatted = layers.Concatenate(axis=1)([normalize, dense1])
        concatted._name = "Join First Stage and ASCII Mean"   
        '''
        visible =  tf.keras.Input(shape=(1), dtype=tf.string)
        
        vectorize = self.vectorize_layer(visible)
        vectorize._name = "Vectorize"
        
        embedding =layers.Embedding(MAX_TOKENS + 1, MIDDLE_LAYER)(vectorize)
        embedding._name = "Embedding"
        
        drop1 =layers.Dropout(0.05)(embedding)
        drop1._name = "First Dropout"
        
        pooling =layers.GlobalAveragePooling1D()(drop1)
        pooling._name = "Pooling"
        
        outputs = layers.Dense(self.languages.shape[0])(pooling)
        
        self.model = tf.keras.Model(inputs = visible, outputs = outputs, name= "WordLanguageClassificationModel")
        
        
        self.model.compile(loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(0.01),
              metrics=tf.metrics.CategoricalAccuracy())   
        
        
if __name__=="__main__":
    print("Pls dont run the module like this")
    sys.exit()
