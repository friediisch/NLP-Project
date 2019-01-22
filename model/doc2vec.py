from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint
import multiprocessing
from pathlib import Path

fp1 = 'C:/Users/Friedi/OneDrive - bwedu/Unidokumente/Extraktion von Fachwortschatz aus Fachtexten/scrapeLREC/data/Doc2Vec/enwiki-latest-pages-articles.xml.bz2'
fp2 = 'D:/NLP/enwiki-latest-pages-articles.xml'
# with open('D:/NLP/enwiki-latest-pages-articles.xml', 'r', encoding)



class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument([c.decode("utf-8") for c in content], [title])


if __name__ == '__main__':
    wiki = WikiCorpus(Path(fp1))
    documents = TaggedWikiDocument(wiki)