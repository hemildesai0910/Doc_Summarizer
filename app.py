from flask import Flask, request, jsonify
import PyPDF2
import re
import json
import requests
import os
import io
import tempfile
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import logging
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Download NLTK data
# nltk.download('all')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize Flask app
app = Flask(__name__)
CORS(app) 
# Initialize NLP models
logger.info("Loading NLP models...")
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
stop_words = set(stopwords.words('english'))
logger.info("NLP models loaded successfully")

# Process text from a document
def process_text(text):
    """Process text and extract information"""
    # Preprocess the text
    processed_text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into sentences
    sentences = sent_tokenize(processed_text)
    
    # Extract information
    summary = generate_summary(processed_text)
    entities = extract_entities(processed_text)
    keywords = extract_keywords(processed_text)
    main_points = extract_main_points(sentences)
    
    # Create result dictionary
    result = {
        "summary": summary,
        "main_points": main_points,
        "keywords": keywords,
        "entities": entities
    }
    
    return result

# Extract text from a PDF file
def extract_text_from_pdf(file_obj):
    """Extract text from a PDF file"""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file_obj)
        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()
            if page_text:  # Check if text was extracted
                text += page_text + "\n"
            else:
                logger.warning(f"No text extracted from page {page_num+1}")
        
        if not text:
            logger.warning("No text extracted from PDF")
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Extract text from a text file
def extract_text_from_text_file(file_obj):
    """Extract text from a text file"""
    try:
        # Reset file pointer to the beginning
        file_obj.seek(0)
        
        # Try to read as UTF-8
        try:
            content = file_obj.read().decode('utf-8')
        except UnicodeDecodeError:
            # Reset file pointer and try Latin-1
            file_obj.seek(0)
            content = file_obj.read().decode('latin-1')
        
        return content
    except Exception as e:
        logger.error(f"Error reading text file: {str(e)}")
        return ""

# Generate a summary
def generate_summary(text):
    """Generate a summary of the text"""
    # Handle long texts by chunking
    max_chunk_length = 1024
    
    # If text is very short, return it as is
    if len(text) < 100:
        return text
    
    try:
        if len(text) > max_chunk_length:
            chunk = text[:max_chunk_length]
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        else:
            summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        # Fallback to first few sentences
        sentences = sent_tokenize(text)
        return " ".join(sentences[:3])

# Extract entities
def extract_entities(text):
    """Extract named entities from the text"""
    # Limit text size to prevent memory issues
    max_text_length = 1000000
    if len(text) > max_text_length:
        text = text[:max_text_length]
        
    doc = nlp(text)
    
    entities = {
        "people": [],
        "organizations": [],
        "locations": [],
        "dates": [],
        "other": []
    }
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["people"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["organizations"].append(ent.text)
        elif ent.label_ == "GPE" or ent.label_ == "LOC":
            entities["locations"].append(ent.text)
        elif ent.label_ == "DATE":
            entities["dates"].append(ent.text)
        else:
            entities["other"].append((ent.text, ent.label_))
    
    # Remove duplicates while preserving order
    for category in entities:
        if category != "other":
            entities[category] = list(dict.fromkeys(entities[category]))
        else:
            seen = set()
            entities[category] = [x for x in entities[category] if x[0] not in seen and not seen.add(x[0])]
    
    return entities

# Extract keywords
def extract_keywords(text):
    """Extract keywords from the text"""
    # Limit text size to prevent memory issues
    max_text_length = 100000
    if len(text) > max_text_length:
        text = text[:max_text_length]
    
    try:
        vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top keywords
        dense = tfidf_matrix.todense()
        phrase_scores = [(score, feature) for feature, score in zip(feature_names, dense[0].tolist()[0])]
        sorted_phrase_scores = sorted(phrase_scores, key=lambda x: x[0], reverse=True)
        keywords = [phrase for score, phrase in sorted_phrase_scores[:15]]
        
        return keywords
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        # Fallback for very short texts
        words = [word for word in text.lower().split() if word not in stop_words]
        return list(set(words))[:15]

# Extract main points
def extract_main_points(sentences):
    """Extract main points from the sentences"""
    if len(sentences) <= 5:
        return sentences
    
    important_indicators = ["important", "significant", "key", "main", "critical",
                          "essential", "crucial", "primary", "major", "fundamental"]
    
    scored_sentences = []
    
    for i, sentence in enumerate(sentences):
        score = 0
        
        # Position score - first sentences often contain important information
        if i < 3:
            score += 3
        elif i < 10:
            score += 1
        
        # Length score - longer sentences often contain more information
        if len(sentence.split()) > 20:
            score += 2
        elif len(sentence.split()) > 10:
            score += 1
        
        # Content score - sentences with important indicators
        for indicator in important_indicators:
            if indicator in sentence.lower():
                score += 2
                break
        
        scored_sentences.append((sentence, score))
    
    # Sort by score and take top sentences
    sorted_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
    main_points = [sentence for sentence, score in sorted_sentences[:7]]
    
    return main_points

# Download and process online file
def download_file_from_url(url):
    """Download and process a file from a URL"""
    try:
        logger.info(f"Downloading file from: {url}")
        
        # Get the filename from the URL
        filename = url.split("/")[-1]
        if not filename:
            filename = "downloaded_file"
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        logger.info(f"File downloaded to temporary location: {temp_path}")
        
        return temp_path, filename
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return None, None

# Process a file and return the analysis
def analyze_file(file_path, filename):
    """Analyze a file and return the analysis as JSON"""
    try:
        # Determine file type
        if filename.lower().endswith('.pdf'):
            with open(file_path, 'rb') as file:
                text = extract_text_from_pdf(file)
        else:
            with open(file_path, 'rb') as file:
                text = extract_text_from_text_file(file)
        
        if not text:
            return {"error": "No text could be extracted from the file"}, 400
        
        # Process the text
        result = process_text(text)
        
        # Add file information
        result["filename"] = filename
        result["processed_date"] = datetime.now().isoformat()
        
        return result, 200
    except Exception as e:
        logger.error(f"Error analyzing file: {str(e)}")
        return {"error": str(e)}, 500

import requests
from urllib.parse import quote

def fetch_github_file(url):
    """Fetch a file from GitHub with improved error handling and URL encoding"""
    try:
        print(f"Original URL: {url}")
        
        # Convert URL if it's a GitHub web URL
        if "github.com" in url and "blob" in url:
            # Convert from web URL to raw URL
            url = url.replace("github.com", "raw.githubusercontent.com")
            url = url.replace("/blob/", "/")
        elif "github.com" in url and "raw.githubusercontent.com" not in url:
            # Handle repository root URL by appending README.md
            if url.endswith("/"):
                url = url + "README.md"
            else:
                url = url + "/README.md"
            url = url.replace("github.com", "raw.githubusercontent.com")
            url = url.replace("/blob/", "/")
        
        # Encode the URL to handle special characters
        encoded_url = quote(url, safe='/:')
        print(f"Converted URL: {encoded_url}")
        
        # Fetch the content
        response = requests.get(encoded_url, timeout=10)
        
        # Print status code for debugging
        print(f"Response status code: {response.status_code}")
        
        # Handle common HTTP errors specifically
        if response.status_code == 404:
            print("File not found (404). Repository or file may not exist or might be private.")
            return None, None
        elif response.status_code == 403:
            print("Access forbidden (403). May be rate limited or repository is private.")
            return None, None
            
        response.raise_for_status()  # Raise an exception for other HTTP errors
        
        # Get the filename from the URL
        filename = url.split("/")[-1]
        
        return response.text, filename
    except requests.exceptions.Timeout:
        print("Request timed out. Check network connectivity.")
        return None, None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        return None, None
    except requests.exceptions.ConnectionError:
        print("Connection error. Check network connectivity and firewall settings.")
        return None, None
    except Exception as e:
        print(f"Error fetching GitHub file: {e}")
        return None, None

# API endpoint for uploading a file
@app.route('/api/analyze/upload', methods=['POST'])
def analyze_uploaded_file():
    """API endpoint for analyzing an uploaded file"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    print(file)
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check file type
    filename = file.filename
    if not (filename.lower().endswith('.pdf') or filename.lower().endswith(('.txt', '.text', '.md'))):
        return jsonify({"error": "Unsupported file type. Only PDF and text files are supported"}), 400
    
    try:
        # Save the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        # Analyze the file
        result, status_code = analyze_file(temp_path, filename)
        
        # Clean up
        os.unlink(temp_path)
        
        return jsonify(result), status_code
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        return jsonify({"error": str(e)}), 500

# API endpoint for analyzing a file from a URL
@app.route('/api/analyze/url', methods=['POST'])
def analyze_url_file():
    """API endpoint for analyzing a file from a URL"""
    data = request.json
    
    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400
    
    url = data['url']
    
    # Check if URL is for GitHub
    if 'github.com' in url:
        return jsonify({"error": "GitHub URLs are not supported by this endpoint. Use /api/analyze/github instead"}), 400
    
    # Download the file
    temp_path, filename = download_file_from_url(url)
    
    if not temp_path or not filename:
        return jsonify({"error": "Failed to download file from URL"}), 400
    
    try:
        # Analyze the file
        result, status_code = analyze_file(temp_path, filename)
        
        # Add source URL
        if status_code == 200:
            result["source_url"] = url
        
        # Clean up
        os.unlink(temp_path)
        
        return jsonify(result), status_code
    except Exception as e:
        # Clean up in case of error
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        logger.error(f"Error processing file from URL: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze/github', methods=['POST'])
def analyze_github_file():
    """Analyze a file from GitHub"""
    data = request.json
    
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400
    
    url = data['url']
    print(f"Received GitHub URL: {url}")
    
    # Fetch the file
    content, filename = fetch_github_file(url)
    
    if not content:
        return jsonify({
            'error': 'Failed to fetch file from GitHub', 
            'details': 'Repository or file may not exist, may be private, or network issues occurred'
        }), 404
    
    if not filename:
        return jsonify({'error': 'Failed to determine filename from URL'}), 400
    
    print(f"Successfully fetched file: {filename} (content length: {len(content)})")
    
    # Process the text
    result = process_text(content)
    
    # Add file information
    result["filename"] = filename
    result["source_url"] = url
    result["processed_date"] = datetime.now().isoformat()
    
    return jsonify(result)



# API endpoint for analyzing text directly
@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    """API endpoint for analyzing text directly"""
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    filename = data.get('filename', 'text_input.txt')
    
    if not text:
        return jsonify({"error": "Empty text provided"}), 400
    
    try:
        # Process the text
        result = process_text(text)
        
        # Add file information
        result["filename"] = filename
        result["processed_date"] = datetime.now().isoformat()
        
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "models_loaded": True}), 200

# Main entry point
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
