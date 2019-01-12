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

# Open a PDF file.
fp = 'data/LREC/LREC2012_Proceedings/pdf/593_Paper.pdf'
# Create a PDF parser object associated with the file object.

def mine_pdf(fp):
    
    with open(fp, 'rb') as file:
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
    for i in lines:
        if i == '':
            i = ' '
        if i[-1] == '-':
            i[-1] = ''
        text += i
    
    
        
    return text