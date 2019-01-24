import gensim
import sys
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import re
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import os
import json
from pathlib import Path



nlp = spacy.load('en_core_web_lg')

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

documents = [] #  [ [token, token, token], [token, token, token], ...]

fp = '../data/LRECjson/'
for jsonfile in os.listdir(Path(fp)):
#for jsonfile in ['../data/LRECjson/2018_1049.json']:
    print(jsonfile)
    with open(Path(str(fp + jsonfile)), 'r') as f:
        try:
            tmp = json.loads(f.read())
            text = tmp['fulltext']
            keywords = tmp['keywords']
            abstract = tmp['abstract']
            title = tmp['title']
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
            text = text.replace(code, value).lower().strip()
            keywords = [kw.replace(code, value).lower().strip() for kw in keywords]
            abstract = abstract.replace(code, value).lower().strip()
            title = title.replace(code, value).lower().strip()
        text = title + abstract + text
        doc = nlp(text)

        # clean:
        doc = [token.lemma_ for token in doc if not token.is_stop and token.shape > 2 and not token.is_currency and not token.is_punct and not token.is_digit]
        documents.append(doc)
        






print("create dict")
dct = Dictionary(documents)
dct.save("../data/models/tfidf/dictionary.model")


print("create tfidf")
model = TfidfModel(dictionary=dct)
model.save("../data/models/tfidf/tfidf.model")

