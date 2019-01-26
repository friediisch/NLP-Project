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

nlp = sp.load('en_core_web_lg')


with open(Path('../data/models/features/data.json'), 'r') as f:
    datalist = json.loads(f.read()) # dictionary:


data = []
labels = []
ngrams = []
for instance_dict in datalist:
    #instance = instance_dict['ngram'] # string
    wv = instance_dict['word_vecs']
    wv = [np.fromstring(vector.strip('[]'), sep=',') for vector in wv]
    ngrams.append(instance_dict['ngram'])
    
    wordvectors = np.array(wv, dtype=np.float32) # [[wv1], [wv2], ...], sorted
    embedding = wordvectors.sum(axis=0) # 300 x 1
    #data.append(embedding)
    data.append(np.fromstring(instance_dict['chunk_vec'].strip('[]'), sep=','))
    labels.append(int(instance_dict['label']))

data = np.array(data)      # shape: (n, 300)
labels = np.array(labels)  # shape: (n, )  
ngrams = np.array(ngrams)


print(data.shape)
print(labels.shape)

print(sum(labels))
positive_examples = sum(labels)

negative_ratio = 2

# sample equal number of negative and positive labels
neg_idx = [i for i in range(labels.shape[0]) if labels[i] == 0] # indices of negative examples
neg_idx = np.random.choice(np.array(neg_idx), positive_examples*negative_ratio, replace=False)

pos_idx = [i for i in range(labels.shape[0]) if labels[i] == 1] # indices of positive examples
pos_idx = np.random.choice(np.array(pos_idx), positive_examples, replace=False)

idx = np.hstack((pos_idx, neg_idx))

labels = labels[idx]
data = data[idx]
ngrams = ngrams[idx]

# print(data.shape)
# print(labels.shape)
# drucker = np.hstack((labels.reshape((-1, 1)), ngrams.reshape((-1, 1))))
# for i in range(drucker.shape[0]):
#     print(drucker[i, :])
#     if i%10==0:
#         sleep(1)

# print(sum(labels))


# split data:
xtrain, xtest, ytrain, ytest, ngrams_train, ngrams_test = train_test_split(data, labels, ngrams, test_size=0.1)


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

for i in range(predictions.shape[0]):
    print(ngrams_test[i])


for i in range(predictions.shape[0]):
    print(predictions[i], '\t', predictions[i] > 0.5 , '\t', ytest[i] )


model.save('../data/model/keras/model{}.h5'.format(str(datetime.now())))



# model = Sequential()
# model.add(Dense(32, input_dim=784))
# model.add(Activation('relu'))

sys.exit(0)

def clean_chunk(chunk):
    
    result = []
    for token in chunk:
        # if token.text.lower() == 'the':
        #     print(token.text.lower().strip(), token.text.lower().strip() in STOP_WORDS)

        if token.text not in STOP_WORDS:
            # if 'brazilian' in chunk.text:
            #     print(token.text)
            #     print(token.is_stop)
            if token.shape > 2:
                if '.' not in token.text.lower():
                    result.append(token.text.lower())
    # if 'brazilian' in chunk.text:
    #     print(chunk.text, '\t', ' '.join(result))        
    result = nlp(' '.join(result).strip(string.punctuation + string.whitespace))
    return result[:]





s = 'this is a text and language processing should be recognized as a keyword'
s = '''
In this paper we describe the Japanese-English Subtitle Corpus (JESC). JESC is a large Japanese-English parallel corpus covering the
underrepresented domain of conversational dialogue. It consists of more than 3.2 million examples, making it the largest freely available
dataset of its kind. The corpus was assembled by crawling and aligning subtitles found on the web. The assembly process incorporates
a number of novel preprocessing elements to ensure high monolingual ﬂuency and accurate bilingual alignments. We summarize its
contents and evaluate its quality using human experts and baseline machine translation (MT) systems.
'''
doc = nlp(s)
chunklist = list(doc.noun_chunks)
chunks = doc.noun_chunks

data = []

for chunk in chunks:
    #print(chunk)
    chunk = clean_chunk(chunk)
    print(chunk)
    data.append(chunk.vector)

data = np.array(data)

predictions = model.predict(data)

print(predictions)

for i, chunk in enumerate(chunklist):
    print(chunk, predictions[i])





