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
from utils import clean_topics


nlp = sp.load('en_core_web_lg')
print(nlp.tokenizer)
print(dir(nlp.tokenizer))
#nlp.tokenizer.token_match = None

with open('stopwords.txt', 'r', encoding='utf-8') as f:
    STOPWORDS = f.readlines()
    STOPWORDS = set([item.strip(string.whitespace) for item in STOPWORDS])
    STOP_WORDS = STOP_WORDS.union(STOPWORDS)


print('the' in STOP_WORDS)
#sys.exit(0)

# encodings:
replace_dict = {
    '\ufb01' : 'fi',
    '\u2019' : '',
    '\u00e9' : 'e', 
    '\u00a8' : '',
    'ﬁ': 'fi',
}




data = [] # [{'ngram':'some text', 'label':1, 
          #   vector:np.array(dx1), vecotrs:np.array(nxd), count: 343, counts:[counts], tfidf:[scores}, 
          # abstract_count:, title_count, document_id: id, document_vector: np.array()}]


# tfidf model
dct = Dictionary.load("../data/models/tfidf/dictionary.model")
tfidf = TfidfModel.load("../data/models/tfidf/tfidf.model")



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







       

# gather all word embeddings in a dictionary
w2v = {}
for token in nlp.vocab:
    if token.has_vector:
        #print(token.text.lower(), token.vector)
        if token.text.lower() not in w2v:    
            w2v[token.text.lower()] = token.vector

#with open('w2v.json', 'w', encoding='utf-8') as fp:
    #json.dump(w2v, fp)
print('got word vectors')


vocab_size = len(w2v)
print(len(nlp.vocab))
print(vocab_size)

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
doc_count = 0
files = os.listdir(Path(fp))
shuffle(files)
for jsonfile in files[:100]:
#for jsonfile in ['../data/LRECjson/2018_1049.json']:
    doc_id = doc_count
    doc_count += 1
    print(jsonfile)
    print(fp, jsonfile)
    print(str(fp + str(jsonfile)))
    path = str(fp + str(jsonfile))
    with open(Path(path), 'r') as f:
    #with open(Path(str(jsonfile)), 'r') as f:
        try:
            tmp = json.loads(f.read())
            text = tmp['fulltext']
            keywords = tmp['keywords'] + clean_topics(tmp['topics'])
            abstract = tmp['abstract']
            title = tmp['title']
            print(keywords) #  ['  blog as corpus', ' blogset-br', ' brazilian portuguese corpu']
            # let the preprocessing begin
            if text is None:
                text = ''
            if keywords is None:
                keywords = []
            if abstract is None:
                abstract = ''
            if title is None:
                title = ''
        except KeyError as e:
            print(e)
            continue
    	
        # replace encoded characters
        for code, value in replace_dict.items():
            #print('replacing broken unicode')
            text = text.replace(code, value).lower().strip()
            keywords = [kw.replace(code, value).lower().strip() for kw in keywords if kw is not None]
            abstract = abstract.replace(code, value).lower().strip()
            title = title.replace(code, value).lower().strip()
        
        if keywords is None or len(keywords) == 0:
            print('skipping...')
            continue

        # create spacy document
        
        
        text_ = nlp(text)
        #print('text is now a spacy object!')
        keywords_ = [nlp(kw.lower().strip()) for kw in keywords]
        #print('keywords are now list of spacy object!')
        abstract_ = nlp(abstract)
        title_ = nlp(title)
        #print('abstract and title are spacy objects')

        
        __text__ = [token.lemma_ for token in text_ if not token.is_stop and token.shape > 2 and not token.is_currency and not token.is_punct and not token.is_digit]
        __title__ = [token.lemma_ for token in title_ if not token.is_stop and token.shape > 2 and not token.is_currency and not token.is_punct and not token.is_digit]
        __abstract__ = [token.lemma_ for token in abstract_ if not token.is_stop and token.shape > 2 and not token.is_currency and not token.is_punct and not token.is_digit]
        
        bow = dct.doc2bow(__title__ + __abstract__ + __text__)
        
        scores = tfidf[bow]  # [(tokenid , tfidf_score), (), ...]
        tfidf_score_dict = dict([(dct[index], weight) for index, weight in scores]) # [(token , tfidf_score), (), ...]


        lemma_dict = {'text': ' '.join(__text__), 'title': ' '.join(__title__), 'abstract': ' '.join(__abstract__)}

        # get document vector:

        title_vec = title_.vector
        abstract_vec = abstract_.vector
        text_vec = text_.vector


        candidate_set = set()
        for body in [title_, abstract_, text_]:
            for chunk in body.noun_chunks:
                chunk = clean_chunk(chunk)
                if len(chunk) > 0:
                    chunk_text = chunk.text.lower().strip()
                    if chunk_text not in candidate_set:
                        candidate_set.add(chunk_text)
                        candidate_dict = {}
                        # check if candidate is a keyword:
                        for keyword in keywords_:
                            #print(chunk.text.lower(), '\t',  keyword.text.lower())
                            keyword_text = keyword.text.lower().strip()
                            if keyword_text == chunk_text or editdistance.eval(chunk_text, keyword_text) <= (len(keyword_text)>4) * (1 * (len(keyword) >1) +  max(1, np.floor(np.log(len(keyword_text))))):
                                label = 1

                                print('checking keywords: ', chunk_text, '\t', keyword_text, 'edit distance: ', editdistance.eval(chunk_text, keyword_text))
                                candidates_kw.append(chunk_text)
                                y_kw.append(label)
                                break
                        else: 
                            #print(chunk.text.lower(), '\t',  keyword.text.lower()) (len(keyword.text.lower())>2) *
                            label = 0
                            candidates_not_kw.append(chunk_text)
                            y_not_kw.append(label)

                        # extract features
                        # todo
                        # tfidf, term frequency, dummy für abstract und title
                        # flexionen von einem Term:
                        # die anzahl variationen von einem speziellen term kommen im text vor? (permormance?)
                        # doc2vec: ähnlichkeit term zu dokument
                        # zips law
                        # [{'ngram':'some text', 'label':1, 
                        #   vector:np.array(dx1), vecotrs:np.array(nxd), count: 343, counts:[counts], tfidf:[scores}, 
                        # abstract_count:, title_count, document_id: id, document_vector: np.array()}]
                        
                        candidate_dict['filename'] = jsonfile
                        candidate_dict['doc_id'] = doc_id

                        candidate_dict['ngram'] = chunk_text
                        candidate_dict['label'] = label

                        # chunk vectors
                        candidate_dict['chunk_vec'] = str(list(chunk.vector))
                        candidate_dict['word_vecs'] = [str(list(token.vector)) for token in chunk]
                        

                        # document vectors
                        candidate_dict['title_vec'] = str(list(title_vec))
                        candidate_dict['abstract_vec'] = str(list(abstract_vec))
                        candidate_dict['text_vec'] = str(list(text_vec))

                        title_counter = Counter(__title__)
                        abstract_counter = Counter(__abstract__)
                        text_counter = Counter(__text__)
                        # single word counts
                        #print('HERE')
                        #print(chunk)
                        #for token in chunk:
                            #print(token, abstract_counter[token.lemma_])
                        candidate_dict['abstract_counts'] = [abstract_counter[token.lemma_] for token in chunk]
                        candidate_dict['text_counts'] = [text_counter[token.lemma_] for token in chunk]
                        candidate_dict['title_counts'] = [title_counter[token.lemma_] for token in chunk]

                        # ngram count:
                        for section_name, section in lemma_dict.items():
                            candidate_dict[section_name + '_ngram_count'] = lemma_dict[section_name].count(chunk.lemma_)

                        # tfidf scores:
                        candidate_dict['tfidf_scores'] = str([tfidf_score_dict[token.lemma_] for token in chunk if token.lemma_ in tfidf_score_dict])

                        data.append(candidate_dict)


#     with open(Path('../data/models/features/data.json'), 'w+') as f:
#         json.dump(data, f)


# with open(Path('../data/models/features/data.json'), 'w+') as f:
#     json.dump(data, f)
                        
                        
# bow = dictionary.doc2bow(process_document(new_doc))

# weights = tfidf[bow] # [(tokenid , score), (), ...]
# print(weights)
# print(dictionary.id2token)
# print(dictionary[0])

# kw = sorted([(dictionary[index], weight) for index, weight in weights], key=operator.itemgetter(1))

# print(kw)                 


sys.exit(0)


# clean candidates, such that i contains 4 non-keywords for each keyword: 
print(jsonfile)
number_of_keywords = len(candidates_kw)
print('number of keywords: ', number_of_keywords)
print('number of non keywords: ', len(candidates_not_kw))
print('keywords: ', candidates_kw)
input('press enter')

idx = np.random.choice(len(candidates_not_kw), number_of_keywords, replace=False)
candidates_not_kw =  [candidates_not_kw[i] for i in idx]
#features_not_kw = features_not_kw[:4*number_of_keywords]
y_not_kw = [y_not_kw[i] for i in idx]


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


print(vocab_size)
encoded_candidates = [one_hot(c, vocab_size) for c in candidates]
print(encoded_candidates)
input('press enter')
#print(encoded_candidates)
# pad documents to a max length of 4 words

max_length = 5
padded_docs = pad_sequences(encoded_candidates, maxlen=max_length, padding='post')
#print(padded_docs)
# define the model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_length))
model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(10,)))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
#print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=20, verbose=1)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=1)
print('Accuracy: %f' % (accuracy*100))




input('press enter')
#predictions = model.predict(padded_docs)
#print(predictions)








# ['blogset - br', 'brazilian portuguese corpus', 'blog corpus', 'brazilian portug
# uese corpora', 'blogset - br', 'blogset - br', 'a brazilian portuguese blog corp
# us', 'a brazilian portuguese blog corpus', 'a brazilian portuguese blog corpus',
#  'internet bloggs', 'internet bloggs', 'internet bloggs', 'scientific communitie
# s', 'scientific communities', 'scientific communities', 'purposes', 'purposes',
# 'purposes', 'opinion', 'opinion']





# https://people.csail.mit.edu/lavanya/keywordfinder



# \ufb01 etc. ersetzen

# stopwords durch token ersetzen
# wörter mit länge 1 raus
# digits durch token
# punctuation durch delimiter
# mail, phone numbers, urls ersetzen
# checken ob Wort im Vokabular


