import spacy as sp
import os
from pathlib import Path
import json
import sys
import string
from spacy.lang.en.stop_words import STOP_WORDS
import numpy as np 
import keras
import editdistance
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding


nlp = sp.load('en_core_web_lg')


# with open('stopwords.txt', 'r', encoding='utf-8') as f:
#     STOPWORDS = f.readlines()
#     STOPWORDS = set([item.strip(string.whitespace) for item in STOPWORDS])
#     STOP_WORDS = STOP_WORDS.union(STOPWORDS)


# encodings:
replace_dict = {
    '\ufb01' : 'fi',
    '\u2019' : '',
    '\u00e9' : 'e', 
    '\u00a8' : '',
    'ﬁ': 'fi',
}

s = 'hahaha\ufb01hhaha\ufb01blabla\ufb01blabal'
s = s.replace('\ufb01', 'fi')
print(s)
# sys.exit(0)
#fulldoc = ''

# fp = 'data/LRECjson/'
# for jsonfile in os.listdir(Path(fp))[:1]:

#     with open(Path(str(fp + jsonfile)), 'r') as f:
#         tmp = json.loads(f.read())
#         text = tmp['fulltext']

#         # let the preprocessing begin

#         # replace encoded characters
#         for code, value in replace_dict.items():
#             text = text.replace(code, value)
        
#         # delete stopwords

        

#         doc = nlp(text)
#         for token in doc:
#             if not nlp.vocab[token.text.lower()].is_oov:
#                 # print(token.text.lower())
#                 # print(nlp.vocab[token.text.lower()].text.lower())
#                 # print(nlp.vocab[token.text.lower()].has_vector)
#                 # print(token.text.lower(), token.idx, token.lemma_, token.pos_, token.shape_, token.tag_)
#             else:
#                 pass
#                 #print(token.text.lower())




def is_head_of_chunk(token):
    if not token.is_stop:
        if not token.is_punct:
            if token.shape_ > 1:
                if not token.is_digit:
                    if not token.is_currency:
                        if not token.like_url:
                            if not token.like_num:
                                if not token.like_email:
                                    return True
    return False


def get_chunks(doc):
    i = 0
    while i < len(doc):
        c = 1
        while True:
            if is_head_of_chunk(doc[i+c-1]):
                c += 1
            else:
                break
        chunk = doc[i:i+c]
        i += c
        yield chunk


def grams(doc, n):
    for chunk in get_chunks(doc):
        for k in range(1, n+1):
            for i in range(len(chunk)):
                yield chunk[i: i+k]






def get_term_length(term):
    return len(term)


def clean_chunk(chunk):
    result = []
    for token in chunk:
        if not token.is_stop:
            if token.shape > 1:
                if '.' not in token.text.lower():
                    result.append(token.text.lower())
    result = nlp(' '.join(result))
    return result


       

# gather all word embeddings in a dictionary
w2v = {}
for token in nlp.vocab:
    if token.has_vector:
        #print(token.text.lower(), token.vector)
        if token.text.lower() not in w2v:
            print(token.vector.shape)
            break      
            w2v[token.text.lower()] = token.vector
#with open('w2v.json', 'w', encoding='utf-8') as fp:
    #json.dump(w2v, fp)
print('got word vectors')

vocab_size = len(w2v)
# create embedding matrix
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in enumerate(nlp.vocab):
	embedding_vector = w2v.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

print('got embedding matrix')

# store data: 
candidates_kw = [] # candidates, string
features_kw = []
y_kw = [] # label, keyword yes no

candidates_not_kw = [] # candidates, string
features_not_kw = []
y_not_kw = [] # label, keyword yes no

fp = '../data/LRECjson/'
for jsonfile in os.listdir(Path(fp))[:1]:

    with open(Path(str(fp + jsonfile)), 'r') as f:
        tmp = json.loads(f.read())
        text = tmp['fulltext']
        keywords = tmp['keywords']
        abstract = tmp['abstract']
        title = tmp['title']
        print(keywords) #  ['  blog as corpus', ' blogset-br', ' brazilian portuguese corpu']
        # let the preprocessing begin

        # replace encoded characters
        for code, value in replace_dict.items():
            text = text.replace(code, value)
            keywords = [kw.replace(code, value) for kw in keywords]
            abstract = abstract.replace(code, value)
            title = title.replace(code, value)
        
        if keywords is None or len(keywords) == 0:
            continue

        # create spacy document
        text_ = nlp(text)
        keywords_ = [nlp(kw) for kw in keywords]
        abstract_ = nlp(abstract)
        title_ = nlp(title)

        candidate_set = set()
        for body in [title_, abstract_, text_]:
            for chunk in body.noun_chunks:
                chunk = clean_chunk(chunk)
                if len(chunk) > 0:
                    if chunk.text.lower() not in candidate_set:
                        candidate_set.add(chunk.text.lower())
                        # check if candidate is a keyword:
                        for keyword in keywords_:
                            #print(chunk.text.lower(), '\t',  keyword.text.lower())
                            if keyword.text.lower() in chunk.text.lower() or editdistance.eval(chunk.text.lower(), keyword.text.lower()) < 1 + min(1, np.floor(np.log(get_term_length(keyword.text.lower())))):
                                label = 1
                                candidates_kw.append(chunk.text.lower())
                                y_kw.append(label)
                            else: 
                                #print(chunk.text.lower(), '\t',  keyword.text.lower())
                                label = 0
                                candidates_not_kw.append(chunk.text.lower())
                                y_not_kw.append(label)

                            # extract features
                            # todo
                            # tfidf, term frequency, dummy für abstract und title
                            # flexionen von einem Term:
                            # die anzahl variationen von einem speziellen term kommen im text vor? (permormance?)
                            # doc2vec: ähnlichkeit term zu dokument
                            # zips law


# clean candidates, such that i contains 4 non-keywords for each keyword: 
print(jsonfile)
number_of_keywords = len(candidates_kw)
print('number of keywords: ', number_of_keywords)
print('number of non keywords: ', len(candidates_not_kw))
print('keywords: ', candidates_kw)
input('press enter')

candidates_not_kw =  candidates_not_kw[:4*number_of_keywords]
features_not_kw = features_not_kw[:4*number_of_keywords]
y_not_kw = y_not_kw[:4*number_of_keywords]


candidates = candidates_kw + candidates_not_kw
y = y_kw + y_not_kw    

print('number of training data: ', len(candidates))
input('press enter')
print(candidates)

input('press enter')

assert len(candidates) == len(y), 'wrong number of labels'
                    
# shuffle
idx = np.random.choice(len(candidates), len(candidates), replace=False)
candidates = [candidates[i] for i in idx]  
y = [y[i] for i in idx]
labels = np.array(y)

encoded_candidates = [one_hot(c, vocab_size) for c in candidates]
#print(encoded_candidates)
# pad documents to a max length of 4 words
max_length = 5
padded_candidates = pad_sequences(encoded_candidates, maxlen=max_length, padding='post')
#print(padded_docs)

            
model = Sequential()
e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=False)
model.add(e)
model.add(Flatten()) # 1500 nodes after flatten
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='relu'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_candidates, labels, epochs=15, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_candidates, labels, verbose=1)
print('Accuracy: %f' % (accuracy*100))









# https://people.csail.mit.edu/lavanya/keywordfinder



# \ufb01 etc. ersetzen

# stopwords durch token ersetzen
# wörter mit länge 1 raus
# digits durch token
# punctuation durch delimiter
# mail, phone numbers, urls ersetzen
# checken ob Wort im Vokabular


