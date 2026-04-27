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
from chromadb_client import client, collection




# Initialize the Sentence Transformer model and ChromaDB client
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# AWS configuration
s3_client = boto3.client('s3')
bucket_name = "hr-api-inoday"  # Replace with your actual S3 bucket name

# AWS Bedrock Client setup for Job Description and Interview Questions generation
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set Tesseract path if necessary
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Utility to extract text based on file type
def extract_text_from_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension in ['.jpg', '.jpeg', '.png']:
        return extract_text_from_image(file_path)
    elif file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension in ['.doc', '.docx']:
        return extract_text_from_doc(file_path)
    else:
        raise ValueError("Unsupported file type. Supported formats: JPG, JPEG, PNG, PDF, DOC, DOCX.")

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


# Load resumes from a JSON file
def load_resumes():
    if os.path.exists("resumes.json"):
        with open("resumes.json", "r") as file:
            return json.load(file)
    return []

# Save resumes to a JSON file
def save_resumes(data):
    with open("resumes.json", "w") as file:
        json.dump(data, file, indent=4)

# Utility to interact with AWS Bedrock
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

# Generate job description based on skills
def generate_job_description(skills):
    prompt = f"Create a detailed job description for the following skills: {skills}"
    return generate_text_with_bedrock(prompt)

# Generate interview questions
def generate_interview_questions(input_text, input_type="Job Description"):
    prompt = f"Based on the following {input_type}, generate 5 interview questions:\n\n{input_text}"
    return generate_text_with_bedrock(prompt)

# Extract key details from resume text
def generate_keypoints_with_bedrock(text):
    prompt = f"Extract the following details from the resume text: name, date of birth, email, phone number.Be concise and precise to the point. Do not give extra information. Here is the resume text:\n\n{text}"
    return generate_text_with_bedrock(prompt)

# Extract skills from resume text
def generate_skills_with_bedrock(text):
    prompt = f"Extract the following details from the resume text: Skills(Such as Technical and Functional Skills).Be concise and precise to the point. Do not give extra information. Here is the resume text:\n\n{text}"
    return generate_text_with_bedrock(prompt)

# Generate embeddings for text
def generate_embeddings_with_hf(text):
    return model.encode(text)

# Process resume and add to ChromaDB
def process_resume(file_path):
    text = extract_text_from_file(file_path)
    keypoints = generate_keypoints_with_bedrock(text)
    skills = generate_skills_with_bedrock(text)
    metadata = {
        "name": keypoints.get('name', ''),
        "email": keypoints.get('email', ''),
        "phone": keypoints.get('phone', ''),
        "dob": keypoints.get('dob', ''),
        "skills": skills,
    }
    embedding = generate_embeddings_with_hf(text)
    collection.add(
        documents=[text],
        metadatas=[metadata],
        embeddings=[embedding],
        ids=[os.path.basename(file_path)]
    )
    resumes = load_resumes()
    resumes.append(metadata)
    save_resumes(resumes)
    return metadata

# Modify upload_resume_to_s3 to automatically process the resume
def upload_resume_to_s3_and_process(file_path, file_name):
    try:
        s3_client.upload_file(file_path, bucket_name, file_name)
        print(f"Uploaded {file_name} to S3 bucket {bucket_name}.")
        metadata = process_resume(file_path)  # Process and update ChromaDB
        return metadata
    except Exception as e:
        print(f"Error uploading to S3 or processing resume: {str(e)}")
        return None

def sync_s3_with_chroma():
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            for obj in response['Contents']:
                file_name = obj['Key']
                local_path = f"/tmp/{file_name}"
                s3_client.download_file(bucket_name, file_name, local_path)
                process_resume(local_path)  # Process and add to ChromaDB
                os.remove(local_path)  # Clean up local file
    except Exception as e:
        print(f"Error syncing S3 with ChromaDB: {str(e)}")

# Rank resumes based on a query
def rank_resumes_based_on_query(query):
    # Generate query embedding
    query_embedding = generate_embeddings_with_hf(query)
    # Query the database and get results
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    # List to store ranked resumes
    ranked_resumes = []
    for doc, metadata, score in zip(results['documents'], results['metadatas'], results['distances']):
        # Calculate similarity score (lower distance => higher similarity)
        similarity_score = round((1 - score) * 100, 2)  # Convert distance to similarity score (percentage)
        
        # Add resume to the ranked list with detailed information
        ranked_resumes.append({
            "name": metadata.get("name", ""),
            "email": metadata.get("email", ""),
            "phone": metadata.get("phone", ""),
            "skills": metadata.get("skills", ""),
            "score": similarity_score  # Return the calculated similarity score
        })
    
    # Return the ranked resumes list
    return ranked_resumes
