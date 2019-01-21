import spacy as sp
import os
from pathlib import Path
import json
import sys
import string
from spacy.lang.en.stop_words import STOP_WORDS


nlp = sp.load('en_core_web_lg')


with open('model/stopwords.txt', 'r', encoding='utf-8') as f:
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

s = 'hahaha\ufb01hhaha\ufb01blabla\ufb01blabal'
s = s.replace('\ufb01', 'fi')
print(s)
# sys.exit(0)
fulldoc = ''

fp = 'data/LRECjson/'
for jsonfile in os.listdir(Path(fp))[:1]:

    with open(Path(str(fp + jsonfile)), 'r') as f:
        tmp = json.loads(f.read())
        text = tmp['fulltext']

        # let the preprocessing begin

        # replace encoded characters
        for code, value in replace_dict.items():
            text = text.replace(code, value)
        
        # delete stopwords

        

        doc = nlp(text)
        for token in doc:
            if not nlp.vocab[token.text].is_oov:
                # print(token.text)
                # print(nlp.vocab[token.text].text)
                # print(nlp.vocab[token.text].has_vector)
                # print(token.text, token.idx, token.lemma_, token.pos_, token.shape_, token.tag_)
                
                
            else:
                pass
                #print(token.text)

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


        





# \ufb01 etc. ersetzen

# stopwords durch token ersetzen
# wörter mit länge 1 raus
# digits durch token
# punctuation durch delimiter
# mail, phone numbers, urls ersetzen
# checken ob Wort im Vokabular


