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

nlp = sp.load('en_core_web_lg')


with open(Path('../data/models/features/data.json'), 'r') as f:
    datalist = json.loads(f.read()) # dictionary:


fulldata, labels, ngrams = utils.create_features(datalist)


positive_examples = sum(labels)

negative_ratio = 2

# sample equal number of negative and positive labels
neg_idx = [i for i in range(labels.shape[0]) if labels[i] == 0] # indices of negative examples
neg_idx = np.random.choice(np.array(neg_idx), positive_examples*negative_ratio, replace=False)

pos_idx = [i for i in range(labels.shape[0]) if labels[i] == 1] # indices of positive examples
pos_idx = np.random.choice(np.array(pos_idx), positive_examples, replace=False)

idx = np.hstack((pos_idx, neg_idx))

labels = labels[idx]
fulldata = fulldata[idx]
ngrams = ngrams[idx]

data1 = fulldata[:,:300]

# split data:
xtrain, xtest, ytrain, ytest, ngrams_train, ngrams_test = train_test_split(data1, labels, ngrams, test_size=0.1)


# define the model
model = Sequential()
model.add(Dense(300, input_dim=300))
model.add(Dense(150, activation='relu'))
model.add(Dense(75, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(xtrain, ytrain, epochs=20, verbose=1)
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
    ytest.reshape((-1,))])



df = pd.DataFrame(df_.T, columns = ['ngram', 'probability', 'label'])


df.to_csv('../data/models/keras/results_model1_2.csv', sep=';')

# for i in range(predictions.shape[0]):
#     print(ngrams_test[i])


# for i in range(predictions.shape[0]):
#     print(predictions[i], '\t', predictions[i] > 0.5 , '\t', ytest[i] )


model.save('../data/models/keras/model1_2.h5')



one_dim_embedding = model.predict(data1).reshape(-1, 1) # n,1
data2 = fulldata[:,300:]  # n x 46
print(data2.shape)
print(one_dim_embedding.shape)
#data2[:, -1] = one_dim_embedding
data2 = np.append(data2, one_dim_embedding, axis=1)
# split data:
xtrain, xtest, ytrain, ytest, ngrams_train, ngrams_test = train_test_split(data2, labels, ngrams, test_size=0.1)





# define the model
model = Sequential()
model.add(Dense(50, input_dim=47))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(xtrain, ytrain, epochs=20, verbose=1)
# evaluate the model
loss, accuracy = model.evaluate(xtest, ytest, verbose=1)
print('Accuracy: %f' % (accuracy*100))

predictions = model.predict(xtest)

# df_ = np.array([ngrams_test.reshape((-1,1)), 
#     predictions.reshape((-1,1)), 
#     ytest.reshape((-1,1)), 
#     ((predictions > 0.5)*1).reshape((-1,1))])



df_ = np.array([ngrams_test.reshape((-1,)), 
    predictions.reshape((-1,)), 
    ytest.reshape((-1,))])



df = pd.DataFrame(df_.T, columns = ['ngram', 'probability', 'label'])


df.to_csv('../data/models/keras/results_model2_2.csv', sep=';')

# for i in range(predictions.shape[0]):
#     print(ngrams_test[i])


# for i in range(predictions.shape[0]):
#     print(predictions[i], '\t', predictions[i] > 0.5 , '\t', ytest[i] )


model.save('../data/models/keras/model2_2.h5')





