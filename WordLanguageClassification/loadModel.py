#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 21:28:44 2023

@author: be
"""
import os
cd = os.path.dirname(os.path.realpath(__file__))
import tensorflow as tf
from LanguageDecoder import LanguageDe

def main():
    tf.get_logger().setLevel('ERROR')
    ld = LanguageDe()

    model = tf.keras.models.load_model(cd + '/model/')
    print("You sucesfully loaded: " + model.name)
    print("Please input a word, I will guess the language! Enter 'quit' to stop the program.")

    while True:
        z = input()
        if z == "quit" or z == "stop" or z == "quit()":
            exit()
        print(ld.decode((model.predict([z]))))
        
if __name__=="__main__":
    main()

