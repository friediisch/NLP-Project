import json
import re
from bs4 import BeautifulSoup
import os
import glob



path = ".\\data\LREC\\"


# for year in [2010, 2012, 2014, 2018]:
#     tmp_path = path + 'LREC{}_Proceedings'.format(year)


#     pdf_files = glob.glob("/home/adam/*.txt")


# path = ".\\data\LREC\\LREC2016_Proceedings\\summaries\\19.html"

# html = open(path, 'r').read()
# soup = BeautifulSoup(html, , 'html.parser')

# soup.find('a', href=)



# data = {
# 'author': 'Philip Kurzendörfer',
# 'year' : 2019,
# 'number' : 10,
# 'title':'dies ist ein sehr sinnloser titel',
# 'abstract': 'Alles was gut ist',
# 'topics': ['topic1', 'topic 2', 'topic 3'],
# 'keywords': ['kw1', 'kw2', 'kw3'],
# 'fulltext': 'dies ist ein sehr kurzer beispiel text ohne nennenswerten inhalt oder sinn sondern einfach sinnlos aneinandergekettete wörter die man problemlos auch ersetzen kann durch andere wörter die dem text mehr sinn geben würden',
# }


# # with open('data.json', 'w') as fp:
# #     json.dump(data, fp)





import io

from pdfminer.converter import TextConverter

from pdfminer.pdfinterp import PDFPageInterpreter

from pdfminer.pdfinterp import PDFResourceManager

from pdfminer.pdfpage import PDFPage

def extract_text_from_pdf(pdf_path):

    resource_manager = PDFResourceManager()

    fake_file_handle = io.StringIO()

    converter = TextConverter(resource_manager, fake_file_handle)

    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    with open(pdf_path, 'rb') as fh:

        for page in PDFPage.get_pages(fh, 

                                      caching=True,

                                      check_extractable=True):

            page_interpreter.process_page(page)

        text = fake_file_handle.getvalue()

    # close open handles

    converter.close()

    fake_file_handle.close()

    if text:

        return text



def extract_text_by_page(pdf_path):

    with open(pdf_path, 'rb') as fh:

        for page in PDFPage.get_pages(fh, 

                                      caching=True,

                                      check_extractable=True):

            resource_manager = PDFResourceManager()

            fake_file_handle = io.StringIO()

            converter = TextConverter(resource_manager, fake_file_handle)

            page_interpreter = PDFPageInterpreter(resource_manager, converter)

            page_interpreter.process_page(page)

            text = fake_file_handle.getvalue()

            yield text

            # close open handles

            converter.close()

            fake_file_handle.close()

def extract_text(pdf_path):

    for page in extract_text_by_page(pdf_path):

        print(page)

        print()

# if __name__ == '__main__':

#     print(extract_text(".\\data\LREC\\LREC2012_Proceedings\\pdf\\194_Paper.pdf"))




if __name__ == '__main__':

    print(extract_text_from_pdf(".\\data\LREC\\LREC2012_Proceedings\\pdf\\194_Paper.pdf"))
