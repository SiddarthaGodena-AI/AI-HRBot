import os
import json
import boto3
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template
from PIL import Image
import pytesseract
import fitz  # PyMuPDF for PDF
from docx import Document  # For DOCX files

# Initialize the Sentence Transformer model and ChromaDB client
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = chromadb.Client()
collection = client.get_or_create_collection(name="resumes")

# AWS configuration
s3_client = boto3.client('s3')
bucket_name = "hr-api-inoday"  # Replace with your actual S3 bucket name

# AWS Bedrock Client setup for Job Description and Interview Questions generation
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set Tesseract path if necessary
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Function to extract text based on file type
def extract_text_from_file(file_path):
    # Get file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension in ['.jpg', '.jpeg', '.png']:
        return extract_text_from_image(file_path)
    elif file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension in ['.doc', '.docx']:
        return extract_text_from_doc(file_path)
    else:
        raise ValueError("Unsupported file type. Only JPG, JPEG, PNG, PDF, DOC, and DOCX are supported.")

# Utility to extract text from image using Tesseract
def extract_text_from_image(image_path):
    # Open the image
    image = Image.open(image_path)
    
    # Convert to grayscale (improves OCR accuracy)
    image = image.convert('L')
    
    # Apply thresholding to improve OCR performance
    image = image.point(lambda p: p > 200 and 255)
    
    # Use Tesseract to extract text from the image
    text = pytesseract.image_to_string(image)
    
    return text

# Utility to extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    text = ""
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    
    # Loop through each page and extract text
    for page in doc:
        text += page.get_text()
    
    return text

# Utility to extract text from DOC or DOCX using python-docx
def extract_text_from_doc(doc_path):
    text = ""
    
    # Open the Word document
    document = Document(doc_path)
    
    # Loop through each paragraph and extract text
    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"
    
    return text

# JSON file for storing resumes
JSON_FILE = "resumes.json"

# Load existing data or create new
def load_resumes():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as file:
            return json.load(file)
    return []

# Save resumes to JSON
def save_resumes(data):
    with open(JSON_FILE, "w") as file:
        json.dump(data, file, indent=4)

# # Extract text from PDFs using pdfplumber (Fine grained control better tham mypdf but more time taking)
# def extract_text_from_pdf(pdf_path):
#     with pdfplumber.open(pdf_path) as pdf:
#         return "\n".join(page.extract_text() for page in pdf.pages)
    

# Utility to call AWS Bedrock for text generation
def generate_text_with_bedrock(prompt, model_id="ai21.j2-ultra-v1"):
    body = {
        "prompt": prompt,
        "maxTokens": 1000,
        "temperature": 0.9,
        "topP": 0.9
    }
    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        response_body = response['body'].read().decode('utf-8')
        response_data = json.loads(response_body)
        return response_data['completions'][0]['data']['text']
    except Exception as e:
        return f"Error generating text: {str(e)}"


# Function to generate job description using skills
def generate_job_description(skills):
    prompt = f"Create a detailed job description for the following skills: {skills}"
    return generate_text_with_bedrock(prompt)

# Function to generate interview questions based on input text
def generate_interview_questions(input_text, input_type="Job Description"):
    prompt = f"Based on the following {input_type}, generate 5 interview questions:\n\n{input_text}"
    return generate_text_with_bedrock(prompt)

def generate_keypoints_with_bedrock(text):
    prompt = f"Extract the following details from the resume text: name, date of birth, email, phone number.Be concise and precise to the point. Do not give extra information. Here is the resume text:\n\n{text}"
    
    body = {
        "prompt": prompt,
        "maxTokens": 8191,
        "temperature": 0.5,
        "topP": 0.9
    }
    
    try:
        response = bedrock_client.invoke_model(
            modelId="ai21.j2-ultra-v1",  # You can change to a model suitable for your task
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        
        response_body = response['body'].read().decode('utf-8')
        response_data = json.loads(response_body)
        return response_data['completions'][0]['data']['text']
    except Exception as e:
        return f"Error generating keypoints: {str(e)}"


def generate_skills_with_bedrock(text):
    prompt = f"Extract the following details from the resume text: Skills(Such as Technical and Functional Skills).Be concise and precise to the point. Do not give extra information. Here is the resume text:\n\n{text}"
    
    body = {
        "prompt": prompt,
        "maxTokens": 8191,
        "temperature": 0.5,
        "topP": 0.9
    }
    
    try:
        response = bedrock_client.invoke_model(
            modelId="ai21.j2-ultra-v1",  # You can change to a model suitable for your task
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        
        response_body = response['body'].read().decode('utf-8')
        response_data = json.loads(response_body)
        return response_data['completions'][0]['data']['text']
    except Exception as e:
        return f"Error generating keypoints: {str(e)}"

from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
def generate_embeddings_with_hf(text):
    embeddings = model.encode(text)
    return embeddings


# def process_resume(file_path):
#     try:
#         # Step 1: Extract text from file
#         extracted_text = extract_text_from_file(file_path)
        
#         # Step 2: Send the extracted text to the model for keypoint extraction
#         keypoints = generate_keypoints_with_bedrock(extracted_text)
#         skills = generate_skills_with_bedrock(extracted_text)

#         # Return or print the extracted keypoints
#         print("Extracted Keypoints:", keypoints)
#         return keypoints
    
#     except Exception as e:
#         return f"Error processing resume: {str(e)}"

# Process a new resume
def process_resume(file_path):
    # Extract text
    if file_path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith((".jpg", ".png")):
        text = extract_text_from_image(file_path)
    else:
        raise ValueError("Unsupported file type.")
    
    # Extract details
    details = generate_keypoints_with_bedrock(text) + generate_skills_with_bedrock(text)
    resume_text = f"{details['name']} {details['email']} {details['phone']} {details['dob']} {' '.join(details['skills'])}"
    
    # Generate embeddings
    embeddings = generate_embeddings_with_hf(text)(resume_text)
    
    # Store details in list and update JSON
    resumes = load_resumes()
    details["embeddings"] = embeddings
    resumes.append(details)
    save_resumes(resumes)
    
    # Add to ChromaDB
    collection.add(
        documents=[resume_text],
        metadatas=[details],
        ids=[f"resume_{len(resumes)}"]
    )
    print(f"Resume processed and stored: {details['name']}")


# Upload resumes to S3
def upload_resume_to_s3(file_path, file_name):
    try:
        s3_client.upload_file(file_path, bucket_name, file_name)
        return True
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        return False

def generate_resume_embedding(resume_text):
    """
    Generate embedding for a given resume text using the sentence transformer model.
    This converts the resume text into a vector (embedding).
    """
    # The 'model.encode' function converts the text into an embedding
    return model.encode(resume_text)

def add_resume_to_chromadb(resume_text, metadata, file_name):
    """
    Add a new resume to ChromaDB, if not already in the database.
    """
    # Generate embedding for the resume
    resume_embedding = generate_resume_embedding(resume_text)

    # Store the resume in ChromaDB
    collection.add(
        documents=[resume_text],
        metadatas=[metadata],
        embeddings=[resume_embedding],
        ids=[file_name]
    )

def rank_resumes_based_on_query(query, ranking_type='skills'):
    """
    Rank resumes in ChromaDB based on a user's query (skills or job role).
    """
    query_embedding = generate_resume_embedding(query)
    
    # Query ChromaDB with the generated embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    ranked_resumes = []
    for result in results['documents']:
        resume_metadata = result['metadata']
        resume_score = cosine_similarity([query_embedding], [result['embedding']])[0][0]
        ranked_resumes.append({
            'file_name': result['id'],
            'skills': resume_metadata.get('skills', ''),
            'name': resume_metadata.get('name', ''),
            'dob': resume_metadata.get('dob', ''),
            'email': resume_metadata.get('email', ''),
            'phone': resume_metadata.get('phone', ''),
            'score': round(resume_score * 100, 2)  # Convert to percentage
        })
    
    return ranked_resumes

# # Add resumes from S3 to ChromaDB
# def add_resumes_from_s3_to_chromadb():
#     response = s3_client.list_objects_v2(Bucket=bucket_name)
#     if "Contents" in response:
#         for obj in response["Contents"]:
#             file_name = obj["Key"]
#             download_path = f"/tmp/{file_name}"
#             s3_client.download_file(bucket_name, file_name, download_path)
            
#             # Determine file type and extract text accordingly
#             if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 resume_text = extract_text_from_image(download_path)
#             elif file_name.lower().endswith('.pdf'):
#                 resume_text = extract_text_from_pdf(download_path)
#             else:
#                 print(f"Unsupported file type: {file_name}")
#                 continue

#             resume_details = extract_resume_details(resume_text)
#             if resume_text.strip():
#                 resume_embedding = model.encode(resume_text)
#                 collection.add(
#                     embeddings=[resume_embedding],
#                     documents=[resume_text],
#                     ids=[file_name],
#                     metadatas=[resume_details]
#     )

#             else:
#                 print(f"Skipped {file_name} due to empty content.")
#     else:
#         print("No files found in S3 bucket.")

# # Rank resumes based on a job description and provide a score
# def rank_resumes(job_description, top_n=5, generate_questions=False):
#     job_embedding = model.encode(job_description)
#     results = collection.query(
#         query_embeddings=[job_embedding],
#         n_results=top_n
#     )
#     output = []
#     for i, (document, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
#         score = model.similarity(job_embedding, model.encode(document))
#         resume_info = {
#             "rank": i,
#             "content": document,
#             "name": metadata.get('name'),
#             "email": metadata.get('email'),
#             "phone": metadata.get('phone'),
#             "skills": metadata.get('skills'),
#             "score": round(score * 100, 2)  # Convert to percentage
#         }

#         # Generate interview questions if requested
#     if generate_questions:
#         resume_info["interview_questions"] = generate_interview_questions(document, input_type="Resume")
        
#         output.append(resume_info)
    
#     return output

# import re

# def extract_resume_details(resume_text):
#     details = {}
#     # Extract email
#     details['email'] = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
#     details['email'] = details['email'].group(0) if details['email'] else None
    
#     # Extract phone number
#     details['phone'] = re.search(r'\b\d{10}\b|\b(?:\d{3}[-.\s]?){2}\d{4}\b', resume_text)
#     details['phone'] = details['phone'].group(0) if details['phone'] else None
    
#     # Extract name using simple heuristic (first line or NER model)
#     lines = resume_text.split('\n')
#     details['name'] = lines[0].strip() if len(lines) > 0 else None

#     # Skills extraction can be rule-based or dictionary-based
#     skills_keywords = ['Python', 'AWS', 'Machine Learning', 'Java', 'C++', 'SQL']  # Extend this list
#     details['skills'] = [skill for skill in skills_keywords if skill.lower() in resume_text.lower()]
    
#     return details
