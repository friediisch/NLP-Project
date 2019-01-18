import gensim
import sys
import codecs
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess
import json
import os
print(gensim.models.doc2vec.FAST_VERSION)
assert gensim.models.doc2vec.FAST_VERSION > -1
from pathlib import Path
from time import time
import string

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tag.stanford import StanfordPOSTagger



# encodings:
replace_dict = {
    '\ufb01' : 'fi',
    '\u2019' : '',
    '\u00e9' : 'e', 
    '\u00a8' : ''
    
}

jarpath = 'C:\daten\data_analytics\VL\computerlinguistic_processes\project\scrapeLREC\data\stanford-postagger-2018-10-16\\stanford-postagger.jar'
modelpath = 'C:\daten\data_analytics\VL\computerlinguistic_processes\project\scrapeLREC\data\stanford-postagger-2018-10-16\\models\\english-bidirectional-distsim.tagger'
wordnet_lemmatizer = WordNetLemmatizer()

tagger = StanfordPOSTagger(modelpath, jarpath)


print(tagger.tag('have'))
sys.exit(0)

with open('stopwords.txt', 'r', encoding='utf-8') as f:
    STOPWORDS = f.readlines()
    STOPWORDS = [item.strip(string.whitespace) for item in STOPWORDS]




def preprocesses_document(text, stopwords=STOPWORDS):
    tokens = word_tokenize(text)














def get_raw_text(textfile):
    with codecs.open(Path(textfile), encoding="utf-8") as f:
        d = json.load(f) 
        text = d['fulltext']
        abstract = d['abstract']
        title = d['title']
        #print(textfile)
        #print('title', title, title is None)
        #print('abstract', abstract, abstract is None)

        raw_text = ""
        for item in [title, abstract, text]:
            if item is not None:
                raw_text = raw_text + ' ' + item
            else:
                print(textfile)
        return text.replace("\"", "\'")


def iter_documentCorpus(dirname):
    """
    Yield processed document:
    @yield: list of tokens
    """

    extracted = 0
    fnames = os.listdir(dirname)

    for file in fnames:
        text = get_raw_text(dirname + file)
        if extracted % 100 == 0:
            print("extracting documentCorpus files: [INFO] extracted: ", extracted)
        yield simple_preprocess(text)
        extracted += 1


class DocumentCorpus(gensim.corpora.TextCorpus):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        i = 0
        for processed_doc in iter_documentCorpus(self.dirname):
            
            yield gensim.models.doc2vec.TaggedDocument(processed_doc, [i])
            i += 1



corpus = DocumentCorpus('../data/LRECjson/')

for text in corpus:
    print(text[0])
    break

sys.exit(0)






doc2vec_dim =100

min_alpha = 0.00001
alpha = 0.0001
num_eras = 2 # number of eras (zeitalter)
num_epochs = 2 # number of epochs per era

alpha_decay = (alpha - min_alpha) / (num_eras)

model = Doc2Vec(vector_size=doc2vec_dim, min_count=10, alpha=alpha, min_alpha=0.00001, dm=0, hs=0, negative=10, window=5)
model.build_vocab(corpus)



start = time()

print("start training")
for era in range(num_eras):
    
    print("now training era ", era)
    model.train(corpus, total_examples=model.corpus_count, epochs=num_epochs)
    model.alpha = max(min_alpha, model.alpha - alpha_decay)
    model.min_alpha = model.alpha
    evaluation = model.wv.accuracy('testcases.txt')
    for section in evaluation:
        print(len(section['correct']) / (len(section['correct']) + len(section['incorrect'])))
    model.save('../data/doc2vec/model')
print(time() - start)

model.save('../data/doc2vec/model')


print(model.docvecs.doctag_syn0.shape)
model = Doc2Vec.load('../data/doc2vec/model')

model.random.seed(1)
vec = model.infer_vector(simple_preprocess("dies ist ein beispiel satz"))
print(vec.shape)


