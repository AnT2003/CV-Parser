from flask import Flask, request, jsonify, send_from_directory
import fitz  # PyMuPDF
import nltk
import spacy
import re
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams
from flask_ngrok import run_with_ngrok  # Import ngrok

# Download NLTK stopwords
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')
nlp2 = spacy.load('output/model-best/')  # Adjust this path to your custom model

app = Flask(__name__)
run_with_ngrok(app)  # Setup the ngrok tunnel

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text

@app.route('/')
def index():
    return send_from_directory('.', 'templates/index.html')

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    file = request.files['file']
    file_path = 'uploaded_pdf.pdf'
    file.save(file_path)
    text = pdf_reader(file_path)
    text = text.strip()
    text = ' '.join(text.split())

    entities = {}

    phone_regex = re.compile(r'''
        (\+?\d{1,3}[\s-]?)?     # Match optional international prefix
        (\(?\d{1,4}\)?[\s-]?)?  # Match optional area code in parentheses
        (\d[\d\s-]{8,13}\d)     # Match between 9 to 15 digits, allowing spaces and dashes
    ''', re.VERBOSE)

    phone_numbers = phone_regex.findall(text)
    for phone_tuple in phone_numbers:
        phone = ''.join(phone_tuple).strip()
        if phone:
            entities.setdefault('PHONE_NUMBER', []).append(phone)

    email_addresses = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in email_addresses:
        entities.setdefault('EMAIL_ADDRESS', []).append(email)

    doc = nlp(text)
    doc2 = nlp2(text)
    
    for ent in doc.ents:
        if ent.label_ in ['EMAIL_ADDRESS', 'PHONE_NUMBER']:
            entities.setdefault(ent.label_, []).append(ent.text)
    
    for ent in doc2.ents:
        if ent.label_.lower() in ['name', 'college name', 'degree', 'skills']:
            entities.setdefault(ent.label_, []).append(ent.text)
    
    return jsonify({'entities': entities})

if __name__ == '__main__':
    app.run()
