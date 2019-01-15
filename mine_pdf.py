#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 20:24:31 2018

@author: Friedemann Schestag
"""
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

# Open a PDF file.
fp = 'data/LREC/LREC2012_Proceedings/pdf/593_Paper.pdf'
# Create a PDF parser object associated with the file object.

def mine_pdf(fp):
    
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
        laparams = LAParams()
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
            
        records = []
        def handle_line(line):
            # Customize your line-by-line parser here
            return records.append(line)
        
        lines = retstr.getvalue().splitlines()
        #for line in lines:
        #    line = handle_line(line)

        text = ""
        recording = False
        got_keywords = False
        for i in range(len(lines)):
            lines[i] = lines[i].lower()
            
            if lines[i].startswith('keywords'):
                got_keywords = True
                keywords = lines[i]
                
                # todo: 
                #   if keywords go over several lines:
                #       uncomment code below to append next lines to keywords
                #        do some outlier management
                #       add rule: if line ends with commata, this indicates that the following line contains further keywords
                counter = i + 1
                while True:
                    if counter > 3:
                        break

                    if lines[counter] != '':
                        keywords += lines[counter]
                        break
                    counter += 1
            
            
                    
            if lines[i] == '':
                lines[i] = ' '
            elif lines[i][-1] == '-':
                lines[i] = lines[i][:-1]
            else:
                lines[i] = lines[i] + ' ' 

            if  got_keywords == True:
                recording = True 

            if recording == True:
                text += lines[i]

            if recording and 'reference' in lines[i] or 'bibliography' in lines[i]:
                if lines[i-1] == '' and lines[i+1] == '':
                    break

        # if text has no keywords, forgett it        
        if got_keywords == False:
            text = None
            keywords = None

        
    return text, keywords