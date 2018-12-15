import codecs
import bs4
import os

os.chdir('/Users/Friedi/OneDrive - bwedu/Unidokumente/Extraktion von Fachwortschatz aus Fachtexten')
years = [2010, 2012, 2014, 2016, 2018]
keyworddict = {}
for year in years:
    summarydir = 'LREC/LREC' + str(year) + '_Proceedings/summaries/'
    os.chdir(summarydir)
    print('searching ' + str(summarydir))
    for summary in os.listdir():
        f=codecs.open(str(summary), 'r')
        try:
            soup = bs4.BeautifulSoup(f, 'html.parser')
        except:
            print(summary + ' not included')
        keywords = []
        for a in bs4.BeautifulSoup(str(soup.find_all(class_='topics_summaries')), 'html.parser').find_all('a'):
            keywords.append(a.string)
        title = soup.find('th', class_="second_summaries").string
        if title in keyworddict.keys():
            print('Doubly assigned title: ' + title)
        keyworddict[title] = keywords
    os.chdir('../../..')