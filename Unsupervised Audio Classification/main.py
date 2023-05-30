#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 08:31:24 2023

This is an unnsupervised ML programn that automatically labels audio data. 


The program works surprisingly well and figures out very pure subclassification.
(Around 93% "pureness accuracy")

"""
# General 
import numpy as np
import pandas as pd
import os

# Tensorflow
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

# Clustering algorithms
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


# Nice but not necessary:
from rich.progress import track

#Get dataFile
if os.path.exists("./datasets") == False:
    print("\033[47mThe program needs to download 1.5GB of audio files, sorry !\033[0m")
    _ = tf.keras.utils.get_file('esc-50.zip',
                            'https://github.com/karoldvl/ESC-50/archive/master.zip',
                            cache_dir='./',
                            cache_subdir='datasets',
                            extract=True)



# Types from the esc dataset that should be classified (animal sounds because 
# they are similar to the acutal problem)
CALLTYPES = ["dog","cat",'pig','crying_baby','hen','siren','crow']



#Transfrom CALLTYPES to Dictionary
map_class_to_id = {}

for i in range(len(CALLTYPES)):
    map_class_to_id[CALLTYPES[i]] = i


#Makes the console look cleaner
tf.compat.v1.logging.set_verbosity(40) # ERROR


#Load Data
esc50_csv = './datasets/ESC-50-master/meta/esc50.csv'
base_data_path = './datasets/ESC-50-master/audio/'

pd_data = pd.read_csv(esc50_csv)
print("Possible Calltypes:")
print(list(pd_data['category'].unique()))

print("\033[32mLoading Yamnet\033[0m")
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. 
    
    This function is copied from a google example
    """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


filtered_pd = pd_data[pd_data.category.isin(CALLTYPES)]

class_id = filtered_pd['category'].apply(lambda name: map_class_to_id[name])
filtered_pd = filtered_pd.assign(target=class_id)

full_path = filtered_pd['filename'].apply(lambda row: os.path.join(base_data_path, row))
filtered_pd = filtered_pd.assign(filename=full_path)

filtered_pd.head(10)

filenames = filtered_pd['filename']

maxNumber = filenames.shape[0]
dataList = []

print("\033[32mTranscoding Data \033[0m")
for i in track(range(maxNumber)):
    dataArray = load_wav_16k_mono(filenames.iat[i])
    scores, embeddings, spectrogram = yamnet_model(dataArray)
    embeddings = tf.reduce_max(embeddings, axis = 0)
    dataList.append(embeddings)


print("\033[32m Preprocessing \033[0m")
X = preprocessing.normalize(np.array(dataList))

pca = PCA(n_components = 35)
Y = pca.fit_transform(X)

print("\033[32m Clustering \033[0m")
cl = AgglomerativeClustering(n_clusters = None, distance_threshold=0.9, compute_full_tree=True, linkage= 'complete').fit(Y)

Nlabels = max(cl.labels_)
CALLTYPES.append("None")
labels = np.zeros((Nlabels + 1,len(CALLTYPES)))

# Count how many elements in each sublabel or from which class
for i in range(len(Y)):
    oneData = filtered_pd['category'].iat[i]
    index = CALLTYPES.index(oneData)
    if index < 0:
        index = len(CALLTYPES)-1 # Last Index
    if cl.labels_[i] == -1:
       cl.labels_[i] = Nlabels - 1
    labels[cl.labels_[i],index] = labels[cl.labels_[i],index] + 1


# Give information about how well the model did
totalError = 0
for i in range(labels.shape[0]):
    row = labels[i,:]
    argMax = row.argmax(axis = 0)
    delta = row.sum() - row[argMax]
    totalError = totalError + delta
    print("Label #" + str(i) + " seems to be part of: " + str(CALLTYPES[argMax])) 
    print("There were " + str(int(100 * delta/row.sum())) + """% impure sublabels in this label\n""")
         
print("There is a total pureness error of: " + str(totalError) + "  out of " + str(len(cl.labels_)) + " samples")
print('''\nThank you for using this program!
      I hope you shall be able to identify many sounds (without doing anything)
             ,.-----__    
          ,:::://///,:::-. 
         /:''/////// ``:::`;/|/
        /'   ||||||     :://'`\ ''' + '''
      .' ,   ||||||     `/(  e \ 
-===~__-'\__X_`````\_____/~`-._ `.
            ~~        ~~       `~-" . .  .  ..     .      .''')
          
          
