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

nlp = sp.load('en_core_web_lg')

with open('stopwords.txt', 'r', encoding='utf-8') as f:
    STOPWORDS = f.readlines()
    STOPWORDS = set([item.strip(string.whitespace) for item in STOPWORDS])
    STOP_WORDS = STOP_WORDS.union(STOPWORDS)


# encodings:
replace_dict = {
    '\ufb01' : 'fi',
    '\u2019' : '',
    '\u00e9' : 'e', 
    '\u00a8' : '',
    'ﬁ': 'fi',
}

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


def clean_topics(topics):
    topics = [topic.split('(')[0].strip() for topic in topics]
    return topics



def read_document(filepath):
    with open(Path(filepath), 'r') as f:
        try:
            tmp = json.loads(f.read())
            title = tmp['title']
            abstract = tmp['abstract']            
            keywords = tmp['keywords'] + clean_topics(tmp['topics'])
            text = tmp['fulltext']
            
            if title is None:
                title = ''   
            if abstract is None:
                abstract = ''        
            if keywords is None:
                keywords = []
            if text is None:
                text = ''


        except KeyError as e:
            print(e)
            return None
    return title, abstract, keywords, text
            




def process_document(title, abstract, keywords, text, doc_id='', jsonfile='', verbose=0):
    # class_balance:
        # None: alle candidates 
        # int: vielfaches der positiven samples die als negative samples ausgegeben werden sollen
    # fuer jedes einzelne chunk:
        # alle features und label

    doc_data = []

    # replace encoded characters
    for code, value in replace_dict.items():
        title = title.replace(code, value).lower().strip()
        abstract = abstract.replace(code, value).lower().strip()
        #keywords = [kw.replace(code, value).lower().strip().split(';') for kw in keywords if kw is not None]
        keywords = [term.replace(code, value).lower().strip(string.whitespace + string.punctuation)  for kw in keywords for term in kw.replace(';', ',').split(',') if kw is not None and term != '']
        
        text = text.replace(code, value).lower().strip()
        
        

    title_ = nlp(title)
    abstract_ = nlp(abstract)   
    keywords_ = [nlp(kw.lower().strip()) for kw in keywords]
    text_ = nlp(text)
    
    
    # overhead for generating tfidf features
    __title__ = [token.lemma_ for token in title_ if not token.is_stop and token.shape > 2 and not token.is_currency and not token.is_punct and not token.is_digit]
    __abstract__ = [token.lemma_ for token in abstract_ if not token.is_stop and token.shape > 2 and not token.is_currency and not token.is_punct and not token.is_digit]
    __text__ = [token.lemma_ for token in text_ if not token.is_stop and token.shape > 2 and not token.is_currency and not token.is_punct and not token.is_digit]
    
    bow = dct.doc2bow(__title__ + __abstract__ + __text__)

    scores = tfidf[bow]  # [(tokenid , tfidf_score), (), ...]
    tfidf_score_dict = dict([(dct[index], weight) for index, weight in scores]) # [(token , tfidf_score), (), ...]

    # overhead ngram counts
    lemma_dict = {'text': ' '.join(__text__), 'title': ' '.join(__title__), 'abstract': ' '.join(__abstract__)}

    title_vec = title_.vector
    abstract_vec = abstract_.vector
    text_vec = text_.vector

    if verbose:
        print('KEYWORDS: ', keywords_)

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
                            if verbose:
                                print('found keyword: ', chunk_text, '\t', 'ED: ', editdistance.eval(chunk_text, keyword_text))

                            break
                    else: 
                        #print(chunk.text.lower(), '\t',  keyword.text.lower()) (len(keyword.text.lower())>2) *
                        label = 0



                    
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

                    doc_data.append(candidate_dict)
    return doc_data # [{dict of chunk1 features}, {dict of chunk2 features}, {}, ...]
                   




def simple_preprocess(text, title='', abstract=''):
    
    
    # replace encoded characters
    for code, value in replace_dict.items():
        title = title.replace(code, value).lower().strip()
        abstract = abstract.replace(code, value).lower().strip()
        text = text.replace(code, value).lower().strip()
        
    title_ = nlp(title)
    abstract_ = nlp(abstract)   
    text_ = nlp(text)

    # data container
    doc_data = []


 # overhead for generating tfidf features
    __title__ = [token.lemma_ for token in title_ if not token.is_stop and token.shape > 2 and not token.is_currency and not token.is_punct and not token.is_digit]
    __abstract__ = [token.lemma_ for token in abstract_ if not token.is_stop and token.shape > 2 and not token.is_currency and not token.is_punct and not token.is_digit]
    __text__ = [token.lemma_ for token in text_ if not token.is_stop and token.shape > 2 and not token.is_currency and not token.is_punct and not token.is_digit]
    
    bow = dct.doc2bow(__title__ + __abstract__ + __text__)

    scores = tfidf[bow]  # [(tokenid , tfidf_score), (), ...]
    tfidf_score_dict = dict([(dct[index], weight) for index, weight in scores]) # [(token , tfidf_score), (), ...]

    # overhead ngram counts
    lemma_dict = {'text': ' '.join(__text__), 'title': ' '.join(__title__), 'abstract': ' '.join(__abstract__)}

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


                    candidate_dict['ngram'] = chunk_text

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
                    candidate_dict['abstract_counts'] = [abstract_counter[token.lemma_] for token in chunk]
                    candidate_dict['text_counts'] = [text_counter[token.lemma_] for token in chunk]
                    candidate_dict['title_counts'] = [title_counter[token.lemma_] for token in chunk]

                    # ngram count:
                    for section_name, section in lemma_dict.items():
                        candidate_dict[section_name + '_ngram_count'] = lemma_dict[section_name].count(chunk.lemma_)

                    # tfidf scores:
                    candidate_dict['tfidf_scores'] = str([tfidf_score_dict[token.lemma_] for token in chunk if token.lemma_ in tfidf_score_dict])

                    doc_data.append(candidate_dict)
    return doc_data # [{dict of chunk1 features}, {dict of chunk2 features}, {}, ...]



def cosim(v1, v2):
    """
    Measures cosine similarity between v1 and v2.
    """
    return np.dot(v1, v2)/(np.sqrt(np.dot(v1, v1))*np.sqrt(np.dot(v2, v2)))


def create_features(data_list, labels=True):
    data = []
    labels = []
    ngrams = []
    pad = 10
    for instance_dict in data_list:

        extract = lambda x: np.fromstring(instance_dict[x].strip('[]'), sep=',')

        # n-gram vector
        chunk_vec = extract('chunk_vec')

        # document section vectors
        title_vec    = extract('title_vec')
        abstract_vec = extract('abstract_vec')
        text_vec     = extract('text_vec')

        # document lemma counts
        title_counts = np.zeros(pad)
        abstract_counts = np.zeros(pad)
        text_counts = np.zeros(pad)

        # title_tmp    = extract('title_counts')
        # abstract_tmp = extract('abstract_counts')
        # text_tmp     = extract('text_counts')
        title_tmp    = np.array(instance_dict['title_counts'])
        abstract_tmp = np.array(instance_dict['abstract_counts'])
        text_tmp     = np.array(instance_dict['text_counts'])

        # get truncation index
        trunc = min(pad, title_tmp.shape[0])

        title_counts[:trunc] = title_tmp[:trunc]
        abstract_counts[:trunc] = abstract_tmp[:trunc]      # alle shape (10,) 
        text_counts[:trunc] = text_tmp[:trunc]

        
        # document n-gram counts
        title_ngram_count    = np.array(instance_dict['title_ngram_count']).reshape(1, )
        abstract_ngram_count = np.array(instance_dict['abstract_ngram_count']).reshape(1, )
        text_ngram_count     = np.array(instance_dict['text_ngram_count']).reshape(1, )

        # n-gram tf-idf score
        tfidf_scores = np.zeros(pad)
        tfidf_tmp = extract('tfidf_scores')
        trunc = min(pad, tfidf_tmp.shape[0])
        tfidf_scores[:trunc] = tfidf_tmp[:trunc]

        # cosine similarities
        title_sim    = np.array(cosim(chunk_vec, title_vec)).reshape(1, )
        abstract_sim = np.array(cosim(chunk_vec, abstract_vec)).reshape(1, )
        text_sim     = np.array(cosim(chunk_vec, text_vec)).reshape(1, )

        # stack features
        features = np.hstack((  chunk_vec,
                                title_counts, abstract_counts, text_counts, 
                                title_ngram_count, abstract_ngram_count, text_ngram_count,
                                tfidf_scores,
                                title_sim, abstract_sim, text_sim))

        # append all features to data
        data.append(features)

        # append label to labels list
        labels.append(int(instance_dict['label']))

        # append ngram to ngram list
        ngrams.append(instance_dict['ngram'])

    # returning X, y, chunks     
    return np.array(data), np.array(labels), np.array(ngrams)
    
