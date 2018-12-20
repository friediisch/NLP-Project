import bs4
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re


path = ".\\data\ACL\\"
url = 'http://aclweb.org/anthology/'
html = urlopen(url).read()

soup = BeautifulSoup(html, 'html.parser')







L = ['J,', 'Q,', 'P,', 'E,', 'N,', 'D,', 'K,', 'S,', 'W,', 'A,', 'C,', 'H,', 'L,', 'Y,', 'O,', 'T']



pattern_old = r'http:\/\/aclweb\.org\/anthology\/[JQPENDKSWACHLYOT]\/[JQPENDKSWACHLYOT]\d\d\/'
pattern = r'[JQPENDKSWACHLYTUX]\/[JQPENDKSWACHLYTUX]\d\d\/'
tags = soup.findAll('a', href = re.compile(pattern))
links = []
for tag in tags:
    print(url + tag.attrs['href'])
    links.append(url + tag.attrs['href'])


def download_file(download_url, name):
    response = urlopen(download_url)
    file = open(path + name, 'wb')
    file.write(response.read())
    file.close()

def get_pdf(links, baseURL=None):
    
        
    pdf_papers = []
    for link in links:
        if baseURL is None:
            baseURL = link
        html = urlopen(link).read()
        soup = BeautifulSoup(html, 'html.parser')
        pattern = r'\.pdf$'
        tags = soup.findAll('a', href = re.compile(pattern))
        
        for tag in tags:
            
            print(baseURL.replace('.html', '/') + tag.attrs['href'])
            pdf_papers.append(baseURL + tag.attrs['href'])
            download_file(baseURL.replace('.html', '/') + tag.attrs['href'], tag.attrs['href'].replace('/', ''))
    return pdf_papers
  

pdfs = get_pdf(links)




endings = ['ANN', 'BIOMED', 'DAT', 'DIAL', 'FSM', 'GEN', 'HAN', 'HUM', 'LEX', 'MEDIA', 'MOL', 'MT', 'NLL', 'PARSE', 'MORPHON', 'SEM', 'SLAV', 'SEMITIC', 'SLPAT', 'UR', 'WAC']
endings = ['sig' + item.lower() for item in endings]

links = []
for ending in endings:
    link = url + ending + '.html'
    links.append(link)
pdfs = get_pdf(links, url)

