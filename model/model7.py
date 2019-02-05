
'''
Class balance auf test set rausgenommen, damit precision nicht verzerrt wird

'''




import spacy as sp
import os
from pathlib import Path
import json
import sys
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Span
import numpy as np 
import keras
import editdistance
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Add
from keras.layers.embeddings import Embedding
from keras.models import load_model
import keras.backend as K
from keras.layers import Lambda
from time import time
import gensim
import sys
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from collections import Counter
from random import shuffle
from time import sleep
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
import utils
import pandas as pd
from keras.layers import Dropout
from keras import regularizers

nlp = sp.load('en_core_web_lg')


with open(Path('../data/models/features/data_3.json'), 'r') as f:
    datalist = json.loads(f.read()) # dictionary:


data, labels, ngrams = utils.create_features(datalist) # fulldata[:300] sind die chunk vectors



docvec = [(
            np.fromstring(instance['title_vec'].strip('[]'), sep=',') +
            np.fromstring(instance['abstract_vec'].strip('[]'), sep=',') + 
            np.fromstring(instance['text_vec'].strip('[]'), sep=',') 
        ) / 3
         for instance in datalist]
docvec = np.array(docvec)


wv = data[:,:300] - docvec # 11 features
tfidf = data[:, 307:308]
data = np.hstack((wv, tfidf))


# split data:
xtrain, xtest, ytrain, ytest, ngrams_train, ngrams_test = train_test_split(data, labels, ngrams, test_size=0.1)







model = load_model('../data/models/keras/model5.h5')

predictions = model.predict(xtest)



df_ = np.array([ngrams_test.reshape((-1,)), 
    predictions.reshape((-1,)), 
    ytest.reshape((-1,)),
    np.where(predictions>0.5, 1, 0).reshape((-1,))])



df = pd.DataFrame(df_.T, columns = ['ngram', 'probability', 'label', 'predicted label'], index=False)


df.to_csv('../data/models/keras/results_model5_modified_classbalance.csv', sep=';')









