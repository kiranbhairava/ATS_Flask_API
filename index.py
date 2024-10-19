import logging
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import spacy
from io import BytesIO
import time

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy NLP model and SentenceTransformer model
try:
    nlp = spacy.load('en_core_web_sm')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("NLP and SentenceTransformer models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        start_time = time.time()
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        end_time = time.time()
        logger.info(f"PDF text extraction completed in {end_time - start_time:.2f} seconds")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

# Function to calculate similarity score between resume and job description
def calculate_similarity(resume_text, job_description_text):
    try:
        start_time = time.time()
        # Embed both resume and job description using SentenceTransformer
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        job_description_embedding = model.encode(job_description_text, convert_to_tensor=True)

        # Compute cosine similarity
        cosine_scores = util.pytorch_cos_sim(resume_embedding, job_description_embedding)
        similarity_score = cosine_scores.item() * 100  # Convert to percentage
        end_time = time.time()
        logger.info(f"Similarity calculation completed in {end_time - start_time:.2f} seconds")
        return similarity_score
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        raise

@app.route('/upload', methods=['POST'])
def upload_resume_and_job_description():
    try:
        start_time = time.time()
        resume_file = request.files.get('resume')
        job_description = request.form.get('job_description')

        logger.info(f"Received resume: {resume_file.filename if resume_file else 'None'}")
        logger.info(f"Received job description length: {len(job_description) if job_description else 0}")

        if not resume_file:
            return jsonify({"error": "No resume file uploaded"}), 400

        if not job_description:
            return jsonify({"error": "No job description provided"}), 400

        # Read and extract text from PDF
        resume_text = extract_text_from_pdf(BytesIO(resume_file.read()))
        
        # Calculate similarity score
        score = calculate_similarity(resume_text, job_description)

        end_time = time.time()
        logger.info(f"Request processed in {end_time - start_time:.2f} seconds")

        return jsonify({"resume_matching_score": f"{score:.2f}%"}), 200
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "An error occurred while processing the request"}), 500

@app.route('/')
def home():
    return jsonify({"message": "Resume Matching API is running!"})
