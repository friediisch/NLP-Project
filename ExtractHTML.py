import codecs
import bs4
import PyPDF2
import os

years = [2010, 2012, 2014, 2016, 2018]
# years = [2012]
datalist = []
for year in years:
    LRECdir = 'data/LREC/LREC' + str(year) + '_Proceedings/'
    print('searching ' + str(LRECdir))
    for summary in os.listdir(LRECdir + 'summaries/'):
        
        data = {}
        
        # open file
        f=codecs.open(str(LRECdir + 'summaries/') + str(summary), 'r')
        try:
            soup = bs4.BeautifulSoup(f, 'html.parser')
        except:
            print(summary + ' not included')
        
        # get year
        data['year'] = year
        
        # get number
        data['number'] = summary.strip('.html')
        
        # extract title
        title = soup.find('th', class_="second_summaries").string
        data['title'] = title

        # extract topics
        topics = []
        for a in bs4.BeautifulSoup(str(soup.find_all(class_='topics_summaries')), 'html.parser').find_all('a'):
            topics.append(a.string)
        data['topics'] = topics
        
        # extract authors
        authors = []
        for a in soup.find('td', text = 'Authors').nextSibling.nextSibling.find_all('a'):
            authors.append(a.string)
        data['authors'] = authors
        
        # extract abstracts
        data['abstract'] = soup.find('td', text = 'Abstract').nextSibling.nextSibling.string
        datalist.append(data)
        
        if year == 2018:
            pdfdir = LRECdir + 'pdf/' + str(data['number']) + '.pdf'
        else:
            pdfdir = LRECdir + 'pdf/' + str(data['number']) + '_Paper.pdf'
        
        pdfFileObj = open(pdfdir, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        num_pages = pdfReader.numPages
        text = ""
        for page_index in range(num_pages):
            pageObj = pdfReader.getPage(page_index)
            text += pageObj.extractText()
        data['fulltext'] = text
            
        

        
        
    

#alltopics = []
#for topiclist in topicsdict:
#    for topic in topiclist.split(' '):
#        alltopics.append(topic)
#alltopics = list(set(alltopics))