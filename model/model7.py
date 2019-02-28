
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
np.random.seed(0)

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

wv = data[:,:300] - docvec
tfidf = data[:, 307:308]
data = np.hstack((wv, tfidf))

res = train_test_split(data, labels, ngrams, wv, tfidf, test_size=0.1, random_state=0)
xtrain, xtest, ytrain, ytest, ngrams_train, ngrams_test, wv_train, wv_test, tfidf_train, tfidf_test = res

train_idx = utils.balance_class(ytrain, negative_ratio=1)
test_idx = utils.balance_class(ytest, negative_ratio=None)

xtrain = xtrain[train_idx]
xtest = xtest[test_idx]
ytrain = ytrain[train_idx]
ytest = ytest[test_idx]
ngrams_train = ngrams_train[train_idx]
ngrams_test = ngrams_test[test_idx]
wv_train = wv_train[train_idx]
wv_test = wv_test[test_idx]
tfidf_train = tfidf_train[train_idx]
tfidf_test = tfidf_test[test_idx]




#########################################################
# Model: w2v -d2v
model1 = load_model('../data/models/keras/model3.h5')

#########################################################
# Model: tfidf
model2 = load_model('../data/models/keras/model4.h5')

#########################################################
# Model: Hybrid
model3 = load_model('../data/models/keras/model5.h5')

# evaluate the models
loss1, accuracy1 = model1.evaluate(wv_test, ytest, verbose=1)
loss2, accuracy2 = model2.evaluate(tfidf_test, ytest, verbose=1)
loss3, accuracy3 = model3.evaluate(xtest, ytest, verbose=1)
predictions1 = model1.predict(wv_test)
predictions2 = model2.predict(tfidf_test)
predictions3 = model3.predict(xtest)

predictions = [predictions1, predictions2, predictions3]

for m in [0, 1, 2]:
    df_ = np.array([ngrams_test.reshape((-1,)), 
        predictions[m].reshape((-1,)), 
        ytest.reshape((-1,)),
        np.where(predictions[m]>0.5, 1, 0).reshape((-1,))])
    df = pd.DataFrame(df_.T, columns = ['ngram', 'probability', 'label', 'predicted label'], index=None)
    df.to_csv('../data/models/keras/results_model{}_modified_classbalance.csv'.format(m+3), sep=';', index=False)









