'''
Nur tfidf scores

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

data = data[:,300:] # 11 features
data = data[:, [7]]


xtrain, xtest, ytrain, ytest, ngrams_train, ngrams_test = train_test_split(data, labels, ngrams, test_size=0.1, random_state=0)

train_idx = utils.balance_class(ytrain, negative_ratio=1)
test_idx = utils.balance_class(ytest, negative_ratio=1)

xtrain = xtrain[train_idx]
xtest = xtest[test_idx]
ytrain = ytrain[train_idx]
ytest = ytest[test_idx]
ngrams_train = ngrams_train[train_idx]
ngrams_test = ngrams_test[test_idx]




# docvec = [(
#             np.fromstring(instance['title_vec'].strip('[]'), sep=',') +
#             np.fromstring(instance['abstract_vec'].strip('[]'), sep=',') + 
#             np.fromstring(instance['text_vec'].strip('[]'), sep=',') 
#         ) / 3
#          for instance in datalist]
# docvec = np.array(docvec)

# positive_examples = sum(labels)

# negative_ratio = 1
# np.random.seed(0)
# # sample equal number of negative and positive labels
# neg_idx = [i for i in range(labels.shape[0]) if labels[i] == 0] # indices of negative examples
# neg_idx = np.random.choice(np.array(neg_idx), positive_examples*negative_ratio, replace=False)

# pos_idx = [i for i in range(labels.shape[0]) if labels[i] == 1] # indices of positive examples
# pos_idx = np.random.choice(np.array(pos_idx), positive_examples, replace=False)

# idx = np.hstack((pos_idx, neg_idx))

# labels = labels[idx]
# data = data[idx]
# ngrams = ngrams[idx]
# #docvec = docvec[idx]

# #data1 = fulldata[:,:300] - docvec

# data = data[:,300:] # 11 features
# data = data[:, [7]]

# print(data.shape)

# # split data:
# xtrain, xtest, ytrain, ytest, ngrams_train, ngrams_test = train_test_split(data, labels, ngrams, test_size=0.1, random_state=0)


# define the model
model = Sequential()
model.add(Dense(1, input_dim=data.shape[1]))
#model.add(Dropout(0.2, input_shape=(data.shape[1],)))

#model.add(Dense(3, activation='relu'))
#model.add(Dropout(0.1, input_shape=(8,)))

#model.add(Dense(3, activation='relu'))
#model.add(Dropout(0.1, input_shape=(5,)))

model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(xtrain, ytrain, epochs=5, verbose=1)
# evaluate the model
loss, accuracy = model.evaluate(xtest, ytest, verbose=1)
print('Accuracy: %f' % (accuracy*100))

predictions = model.predict(xtest)

# df_ = np.array([ngrams_test.reshape((-1,1)), 
#     predictions.reshape((-1,1)), 
#     ytest.reshape((-1,1)), 
#     ((predictions > 0.5)*1).reshape((-1,1))])

print(ytest.shape)
print(ngrams_test.shape)
print(predictions.shape)

df_ = np.array([ngrams_test.reshape((-1,)), 
    predictions.reshape((-1,)), 
    ytest.reshape((-1,)),
    np.where(predictions>0.5, 1, 0).reshape((-1,))])



df = pd.DataFrame(df_.T, columns = ['ngram', 'probability', 'label', 'predicted label'], index=None)


df.to_csv('../data/models/keras/results_model4.csv', sep=';', index=False)

# for i in range(predictions.shape[0]):
#     print(ngrams_test[i])


# for i in range(predictions.shape[0]):
#     print(predictions[i], '\t', predictions[i] > 0.5 , '\t', ytest[i] )


model.save('../data/models/keras/model4.h5')





