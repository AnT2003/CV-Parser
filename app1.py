from flask import Flask, request, jsonify, send_from_directory
import fitz  # PyMuPDF
import nltk
import spacy
from nltk.corpus import stopwords
from spacy import displacy
import re
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams
import pandas as pd
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Download NLTK stopwords
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()
    return text

def extract_degrees(text):
    degrees = []
    keywords = [
        'bachelor', 'master', 'phd', 'doctorate', 'associate', 'diploma', 'degree', 'bsc', 'msc', 'ba', 'ma','mba'
        'cử nhân', 'thạc sĩ', 'tiến sĩ', 'cao đẳng', 'trung cấp', 'bằng đại học', 'bằng cao học'
    ]
    
    # Split text by \n using re.split to preserve line breaks
    lines = re.split(r'\n', text)
    
    for line in lines:
        found = False
        for keyword in keywords:
            if keyword in line.lower():
                # Find the starting index of the keyword
                start_index = line.lower().find(keyword)
                
                # Find the end index before \n character
                end_index = line.find('\n', start_index)
                if end_index == -1:
                    end_index = len(line)
                
                # Extract the degree information
                degree_info = line[start_index:end_index].strip()
                degrees.append(degree_info)
                found = True
                break
        
        if found:
            break  # Stop after finding the first match
    
    return degrees


def extract_college_names(text):
    college_names = []
    keywords = [
        'university', 'college', 'institute', 'academy', 'school', 'polytechnic', 
        'học viện', 'đại học', 'cao đẳng', 'trường đại học', 'trường cao đẳng', 'trường thpt'
    ]

    # Split text by \n using re.split to preserve line breaks
    lines = re.split(r'\n', text)

    for line in lines:
        found = False
        for keyword in keywords:
            if keyword in line.lower():
                # Find the starting index of the keyword
                start_index = line.lower().find(keyword)

                # Find the end index before \n character
                end_index = line.find('\n', start_index)
                if end_index == -1:
                    end_index = len(line)

                # Extract the college name
                college_name = line[start_index:end_index].strip()
                college_names.append(college_name)
                found = True
                break

        if found:
            break  # Stop after finding the first match

    return college_names


@app.route('/')
def index():
    return send_from_directory('.', 'templates/index.html')

@app.route('/process_file', methods=['POST'])
def process_file():
    # Get file from request
    file = request.files['file']

    # Save the file temporarily
    file_path = '/tmp/uploaded_file'
    file.save(file_path)

    # Determine file type and read text accordingly
    if file.filename.endswith('.pdf'):
        text = pdf_reader(file_path)
    else:
        return jsonify({'error': 'Unsupported file format'}), 400

    # Process the text to extract entities
    entities = []
    phone_regex = re.compile(r'''
        # Match international prefix and format variations
        (\+?\d{1,3}[\s-]?)?     # Match optional international prefix
        (\(?\d{1,4}\)?[\s-]?)?  # Match optional area code in parentheses
        (\d[\d\s-]{7,13}\d)     # Match between 9 to 15 digits, allowing spaces and dashes
    ''', re.VERBOSE)

    phone_numbers = phone_regex.findall(text)
    for phone_tuple in phone_numbers:
        phone = ''.join(phone_tuple).strip()
        if phone and (phone.count('-') < 1 or phone.count('-') >1):  # Ensure no single hyphen exists
            entities.append(('Phone Number',phone))

    # Extract email addresses using regex
    email_addresses = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in email_addresses:
        entities.append(('Email Address', email))
    doc = nlp(text)
    nlp2 = spacy.load('output/model-best')
    doc2 = nlp2(text)

    for ent in doc2.ents:
        if ent.label_.lower() in ['name','skills']:
            entities.append((ent.label_,ent.text))

    # Extract degrees using predefined keywords
    degrees = extract_degrees(text)
    for degree in degrees:
        entities.append(('Degree', degree))

    # Extract college names using predefined keywords
    college_names = extract_college_names(text)
    for college_name in college_names:
        entities.append(('College Name', college_name))

    # Convert entities to DataFrame
    df = pd.DataFrame(entities, columns=['info', 'result'])

    # Convert DataFrame to JSON
    json_result = df.to_json(orient='records')

    return json_result

if __name__ == '__main__':
    app.run()
