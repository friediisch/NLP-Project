import bs4
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re

url = 'http://aclweb.org/anthology/'
html = urlopen(url).read()

soup = BeautifulSoup(html, 'html.parser')







L = ['J,', 'Q,', 'P,', 'E,', 'N,', 'D,', 'K,', 'S,', 'W,', 'A,', 'C,', 'H,', 'L,', 'Y,', 'O,', 'T']



pattern_old = r'http:\/\/aclweb\.org\/anthology\/[JQPENDKSWACHLYOT]\/[JQPENDKSWACHLYOT]\d\d\/'
pattern = r'[JQPENDKSWACHLYOT]\/[JQPENDKSWACHLYOT]\d\d\/'

tags = soup.findAll('a', href = re.compile(pattern))
links = []

for tag in tags:
    print(url + tag.attrs['href'])
    links.append(url + tag.attrs['href'])


def get_pdf(links):
    pdf_papers = []
    for link in links:
        html = urlopen(link).read()
        soup = BeautifulSoup(html, 'html.parser')
        pattern = r'\.pdf$'
        tags = soup.findAll('a', href = re.compile(pattern))
        
        for tag in tags:
            pdf_papers.append(link + tag.attrs['href'])
    return pdf_papers

pdfs = get_pdf(links)

endings = ['ANN', 'BIOMED', 'DAT', 'DIAL', 'FSM', 'GEN', 'HAN', 'HUM', 'LEX', 'MEDIA', 'MOL', 'MT', 'NLL', 'PARSE', 'MORPHON', 'SEM', 'SLAV', 'SEMITIC', 'SLPAT', 'UR', 'WAC']
endings = ['sig' + item.lower() for item in endings]



