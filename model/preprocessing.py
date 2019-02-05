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
import utils
from datetime import datetime




data = []

fp = '../data/LRECjson/'
doc_count = 0
files = os.listdir(Path(fp))
start = datetime.now()
shuffle(files)
for jsonfile in files:
#for jsonfile in ['../data/LRECjson/2018_1049.json']:
    doc_id = doc_count
    doc_count += 1
    path = str(fp + str(jsonfile))

    title, abstract, keywords, text = utils.read_document(path)
    
    if None in [title, abstract, keywords, text]:
        continue
    doc_data = utils.process_document(title, abstract, keywords, text, doc_id = doc_id, jsonfile= jsonfile, verbose=1)

    if doc_data is None:
        continue

    # downsample document ngram data
    # [{dict of chunk1 features}, {dict of chunk2 features}, {}, ...]
    labels = [int(instance['label']) for instance in doc_data]

    positive_examples = sum(labels)

    negative_ratio = 10

    if positive_examples > 0:

        # sample fix ratio of negative and positive labels
        neg_idx = [i for i in range(len(labels)) if labels[i] == 0] # indices of negative examples
        neg_idx = np.random.choice(np.array(neg_idx),
                                    min(positive_examples*negative_ratio, len(neg_idx)), 
                                    replace=False)

        pos_idx = [i for i in range(len(labels)) if labels[i] == 1] # indices of positive examples
        pos_idx = np.random.choice(np.array(pos_idx), positive_examples, replace=False)

        idx = np.hstack((pos_idx, neg_idx))

        doc_data = [doc_data[i] for i in idx]

        data += doc_data

        print('Progress: ', str(np.round(doc_count/len(files), 4)), '%')
        print('ETA: ', str(((datetime.now()-start)/doc_count)*(len(files)-doc_count)))



#     # write temporary feature file
#     # with open(Path('../data/models/features/data_2_tmp.json'), 'w+') as f:
#     #     json.dump(data, f)



# # class balancing:
# labels = [int(instance['label']) for instance in data]

# positive_examples = sum(labels)

# negative_ratio = 25

# # sample fix ratio of negative and positive labels
# neg_idx = [i for i in range(len(labels)) if labels[i] == 0] # indices of negative examples
# neg_idx = np.random.choice(np.array(neg_idx),
#                             positive_examples*negative_ratio, 
#                             replace=False)

# pos_idx = [i for i in range(len(labels)) if labels[i] == 1] # indices of positive examples
# pos_idx = np.random.choice(np.array(pos_idx), positive_examples, replace=False)

# idx = np.hstack((pos_idx, neg_idx))

# data = [data[i] for i in idx]


with open(Path('../data/models/features/data_3.json'), 'w+') as f:
    json.dump(data, f)
                        


