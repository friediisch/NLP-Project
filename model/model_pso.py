from PSO import myPSO
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





nlp = sp.load('en_core_web_lg')


with open(Path('../data/models/features/data_2.json'), 'r') as f:
    datalist = json.loads(f.read()) # dictionary:


fulldata, labels, ngrams = utils.create_features(datalist) # fulldata[:300] sind die chunk vectors



docvec = [(
            np.fromstring(instance['title_vec'].strip('[]'), sep=',') +
            np.fromstring(instance['abstract_vec'].strip('[]'), sep=',') + 
            np.fromstring(instance['text_vec'].strip('[]'), sep=',') 
        ) / 3
         for instance in datalist]
docvec = np.array(docvec)

positive_examples = sum(labels)

negative_ratio = 1

# sample equal number of negative and positive labels
neg_idx = [i for i in range(labels.shape[0]) if labels[i] == 0] # indices of negative examples
neg_idx = np.random.choice(np.array(neg_idx), positive_examples*negative_ratio, replace=False)

pos_idx = [i for i in range(labels.shape[0]) if labels[i] == 1] # indices of positive examples
pos_idx = np.random.choice(np.array(pos_idx), positive_examples, replace=False)

idx = np.hstack((pos_idx, neg_idx))

labels = labels[idx]
fulldata = fulldata[idx]
ngrams = ngrams[idx]
docvec = docvec[idx]

#data1 = fulldata[:,:300] - docvec


data1 = fulldata[:,300:] # 11 features
data1 = data1[:, [3, 7, 8, 9, 10]]


# split data:
xtrain, xtest, ytrain, ytest, ngrams_train, ngrams_test = train_test_split(data1, labels, ngrams, test_size=0.1)



# PSO
models = {}
i = 0
def ann_model(num_layers, layer_dims, reg, dropout ,X, X_test, y, y_test):
    num_layers = int(num_layers)
    layer_dims = int(layer_dims)

    model = Sequential()
    model.add(Dense(X.shape[1], input_dim=X.shape[1], kernel_regularizer=regularizers.l2(reg)))
    model.add(Dropout(dropout, input_shape=(X.shape[1],)))

    for _ in range(num_layers-1):
        model.add(Dense(layer_dims, activation='relu', kernel_regularizer=regularizers.l2(reg)))
        model.add(Dropout(dropout, input_shape=(layer_dims,)))
        

    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(reg)))

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    model.fit(X, y, epochs=3, verbose=1)

    score = model.evaluate(X_test, y_test, verbose=1)    
    #print('Accuracy: %f' % (accuracy*100))

    global i
    models[str(i)] = {"model" : model, "num_layers" : num_layers, "layer_dims" : layer_dims, "cost" : score[0], "accuracy" : score[1]}
    i += 1

    print(score)
    return score[0] # cost



def costFunction(x):
    num_layers = int(x[0])
    layer_dims = int(x[1])
    reg =x[2]
    dropout = x[3]

    cost = ann_model(num_layers, layer_dims, reg, dropout, xtrain, xtest, ytrain, ytest)
    return cost




problem = myPSO.OptimizationProblem(costFunction=costFunction, varNames=["num_layers", "layer_dims", 'regularization', 'dropout'], 
                                    nVar=4, varMin=[2, 10, 0, 0], varMax=[20, 10, 2, 0.8])
pso = myPSO.PSO(problem, MaxIter=2, PopSize=3, c1=2, c2=2, w = 2)
g = pso.optimize()

solution = pso.get_solution()
print(solution)

layer_dims = int(solution['layer_dims'])
num_layers = int(solution['num_layers'])
reg = solution['regularization']
dropout = solution['dropout']

# get best model:
modelIndex = None
for i, model_dict in models.items():
    if model_dict["cost"] == g["cost"] and model_dict["num_layers"] == num_layers and model_dict["layer_dims"] == layer_dims and model_dict['regularization'] == reg:

        print("solution found")
        modelIndex = i
        break
else:
    assert False, "An error occurred: No match"

print(models)
model = models[str(modelIndex)]["model"]
score = model.evaluate(xtest, ytest)
print(score)

    












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



df = pd.DataFrame(df_.T, columns = ['ngram', 'probability', 'label', 'predicted label'])


df.to_csv('../data/models/keras/results_modelPSO.csv', sep=';')

# for i in range(predictions.shape[0]):
#     print(ngrams_test[i])


# for i in range(predictions.shape[0]):
#     print(predictions[i], '\t', predictions[i] > 0.5 , '\t', ytest[i] )


model.save('../data/models/keras/modelPSO.h5')



