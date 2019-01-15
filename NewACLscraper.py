import bs4
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import sys
import json
import os
from io import StringIO
print(os.getcwd())
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
# from pdfminer.pdfdevice import PDFDevice
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter

from pathlib import Path

#path = ".\\data\ACL\\"

path = "data/ACL/"
url = 'http://aclweb.org/anthology/'
html = urlopen(url).read()
namecount = 1

soup = BeautifulSoup(html, 'html.parser')







L = ['J,', 'Q,', 'P,', 'E,', 'N,', 'D,', 'K,', 'S,', 'W,', 'A,', 'C,', 'H,', 'L,', 'Y,', 'O,', 'T']



pattern_old = r'http:\/\/aclweb\.org\/anthology\/[JQPENDKSWACHLYOT]\/[JQPENDKSWACHLYOT]\d\d\/'
pattern = r'[JQPENDKSWACHLYTUX]\/[JQPENDKSWACHLYTUX]\d\d\/'
tags = soup.findAll('a', href = re.compile(pattern))
links = []
for tag in tags:
    print(url + tag.attrs['href'])
    links.append(url + tag.attrs['href'])


#https://github.com/obeattie/pdfminer/wiki/pdfminer.layout
def mine_pdf(fp):
    print('mining pdf')
    with open(Path(fp), 'rb') as file:
        parser = PDFParser(file)
    
        # Create a PDF document object that stores the document structure.
        # Supply the password for initialization.
        document = PDFDocument(parser)
        # Check if the document allows text extraction. If not, abort.
        if not document.is_extractable:
            raise PDFTextExtractionNotAllowed
        # Create a PDF resource manager object that stores shared resources.
        rsrcmgr = PDFResourceManager()
        # Create a buffer for the parsed text
        retstr = StringIO()
        
        # Spacing parameters for parsing
        #https://github.com/obeattie/pdfminer/wiki/pdfminer.layout
        laparams = LAParams(char_margin=4.0, word_margin=0)
        #print(laparams.__dict__)
        codec = 'utf-8'
        
        # Create a PDF device object
        device = TextConverter(rsrcmgr, retstr, 
                            codec = codec, 
                            laparams = laparams)
        
        # Create a PDF interpreter object.
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        # Process each page contained in the document.
        for page in PDFPage.create_pages(document):
            interpreter.process_page(page) 
        
        lines = retstr.getvalue().splitlines()

        text = ""
        print('iterate over lines')
        for i in range(len(lines)):
            
            lines[i] = lines[i].lower()
            

            if lines[i] == '':
                lines[i] = ' '
            elif lines[i][-1] == '-':
                lines[i] = lines[i][:-1]
                
            else:
                lines[i] = lines[i] + ' '

            text += lines[i]
    return text




def download_file(download_url):
    print('download file now')
    global namecount
    response = urlopen(download_url)
    file = open(Path('tmp.pdf'), 'wb')
    file.write(response.read())
    file.close()
    fp = 'tmp.pdf'
    text = mine_pdf(fp)
    with open(Path(path + str(namecount) + '.txt'), 'w') as f:
        namecount += 1
        f.write(text)
        

def get_pdf(links, baseURL=None):
    count = 0
        
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
            count += 1
            #print(baseURL.replace('.html', '/') + tag.attrs['href'])
            pdf_papers.append(baseURL + tag.attrs['href'])
            try:
                download_file(baseURL.replace('.html', '/') + tag.attrs['href'])
            except Exception as e:
                print('got an error: ', e)
                pass
            
    return pdf_papers, count
  

pdfs, count1 = get_pdf(links)




endings = ['ANN', 'BIOMED', 'DAT', 'DIAL', 'FSM', 'GEN', 'HAN', 'HUM', 'LEX', 'MEDIA', 'MOL', 'MT', 'NLL', 'PARSE', 'MORPHON', 'SEM', 'SLAV', 'SEMITIC', 'SLPAT', 'UR', 'WAC']
endings = ['sig' + item.lower() for item in endings]

links = []
for ending in endings:
    link = url + ending + '.html'
    links.append(link)
pdfs, count2 = get_pdf(links, url)


print('number of pdf files: ', count1, count2, count1 + count2)
