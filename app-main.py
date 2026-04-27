import os
import json
import boto3
# import chromadb
# from chromadb import Client
# from chromadb.config import Settings
from typing import List
import pytesseract
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import fitz  # PyMuPDF for PDF
import requests
from docx import Document
from typing import Optional
# from neww import (  # Import your utility functions
#     generate_job_description,
#     generate_interview_questions,
#     upload_resume_to_s3_and_process,
#     rank_resumes_based_on_query
# )
from uuid import uuid4
import chromadb
from chromadb import Client
from chromadb.config import Settings

client = chromadb.PersistentClient(path="E:\Inoday\HRBot\save_data")
collection = client.get_or_create_collection(name="resumes")

# Initialize the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# client = chromadb.Client()
# collection = client.get_or_create_collection(name="resumes")


# AWS configuration
s3_client = boto3.client('s3')
bucket_name = "hr-api-inoday"  # Replace with your actual S3 bucket name

# AWS Bedrock Client setup for Job Description and Interview Questions generation
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set Tesseract path if necessary
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


app = FastAPI()

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust according to your requirements
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models for structured input
class JobDescriptionRequest(BaseModel):
    skills: List[str]

class InterviewQuestionsRequest(BaseModel):
    input_text: str
    input_type: str = "Job Description"

class RankResumesRequest(BaseModel):
    job_description: str
    generate_questions: Optional[bool] = False

class ResumeMetadata(BaseModel):
    skills: List[str]
    name: str
    dob: str
    email: str
    phone: str

@app.post("/generate-job-description")
async def generate_job_description_route(request: JobDescriptionRequest):
    try:
        job_description = generate_job_description(request.skills)
        return {"job_description": job_description}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating job description: {str(e)}")


@app.post("/generate-interview-questions")
async def generate_interview_questions_route(request: InterviewQuestionsRequest):
    try:
        interview_questions = generate_interview_questions(request.input_text, request.input_type)
        return {"interview_questions": interview_questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating interview questions: {str(e)}")

# @app.post("/upload-resume")
# async def upload_resume(file: UploadFile = File(...)):
#     try:
#         file_path = f"./{file.filename}"
#         with open(file_path, "wb") as f:
#             content = await file.read()
#             f.write(content)

#         # Upload the file to S3 and process it
#         metadata = upload_resume_to_s3_and_process(file_path, file.filename)
#         os.remove(file_path)  # Clean up local file after processing

#         return {"message": "File uploaded and processed successfully", "metadata": metadata}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error uploading or processing the resume: {str(e)}")

# @app.post("/rank-resumes")
# async def rank_resumes_api(query: str):
#     try:
#         ranked_resumes = rank_resumes_based_on_query(query)
#         return {"ranked_resumes": ranked_resumes}
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"detail": str(e)})

# API Endpoints
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    try:
        file_path = f"./{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        metadata = process_resume(file_path)
        os.remove(file_path)
        return {"message": "Resume uploaded and processed successfully.", "metadata": metadata}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")


@app.post("/rank-resumes")
async def rank_resumes_api(request: dict = Body(...)):
    job_description = request.get("query", "")
    if not job_description:
        return {"error": "Job description (query) is required"}

    # Call your function to rank resumes
    ranked_resumes = rank_resumes_based_on_query(job_description)
    return {"ranked_resumes": ranked_resumes}


    

@app.get("/get-all-resumes")
async def get_all_resumes():
    try:
        # Assuming you're using a ChromaDB collection, if needed you can modify it for other types
        results = collection.query(query_embeddings=[], n_results=5)  # Empty query to fetch all documents
        resumes = []
        for result in results['documents']:
            resume_metadata = result['metadata']
            resumes.append({
                'file_name': result['id'],
                'skills': resume_metadata.get('skills', ''),
                'name': resume_metadata.get('name', ''),
                'dob': resume_metadata.get('dob', ''),
                'email': resume_metadata.get('email', ''),
                'phone': resume_metadata.get('phone', ''),
            })

        return {"resumes": resumes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching resumes: {str(e)}")


@app.get("/")
async def read_root():
    return {"message": "FastAPI is running. Use /docs for API documentation."}




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
        "temperature": 0.2,
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

# def process_resume(file_path: str) -> dict:
#     text = extract_text_from_file(file_path)
#     embedding = model.encode(text)
#     metadata = {
#         "name": f"Resume-{uuid4()}",
#         "skills": "Extracted Skills Placeholder",
#         "email": "email@example.com",
#         "phone": "1234567890",
#         "dob": "1990-01-01"
#     }
#     collection.add(
#         documents=[text],
#         metadatas=[metadata],
#         embeddings=[embedding],
#         ids=[str(uuid4())]
#     )
#     return metadata


def rank_resumes(query: str) -> List[dict]:
    query_embedding = model.encode(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    ranked = []
    for doc, metadata, dist in zip(results['documents'], results['metadatas'], results['distances']):
        score = round((1 - dist) * 100, 2)
        ranked.append({
            "name": metadata.get("name", ""),
            "email": metadata.get("email", ""),
            "phone": metadata.get("phone", ""),
            "dob": metadata.get("dob", ""),
            "skills": metadata.get("skills", ""),
            "score": score
        })
    return ranked






# # Process resume and add to ChromaDB
# def process_resume(file_path):
#     text = extract_text_from_file(file_path)
#     keypoints = generate_keypoints_with_bedrock(text)
#     skills = generate_skills_with_bedrock(text)
#     metadata = {
#         "name": keypoints.get('name', ''),
#         "email": keypoints.get('email', ''),
#         "phone": keypoints.get('phone', ''),
#         "dob": keypoints.get('dob', ''),
#         "skills": skills,
#     }
#     embedding = generate_embeddings_with_hf(text)
#     collection.add(
#         documents=[text],
#         metadatas=[metadata],
#         embeddings=[embedding],
#         ids=[os.path.basename(file_path)]
#     )
#     resumes = load_resumes()
#     resumes.append(metadata)
#     save_resumes(resumes)
#     return metadata

# Process the resume, extract details, and store in ChromaDB
def process_resume(file_path):
    text = extract_text_from_file(file_path)
    resume_details = generate_keypoints_with_bedrock(text) + generate_skills_with_bedrock(text)

    # Generate embedding for the resume
    embedding = generate_embeddings_with_hf(text)
    
    # Add document and metadata to ChromaDB
    collection.add(
        documents=[text],
        metadata=[resume_details],
        embeddings=[embedding],
        ids=[str(uuid4())]  # Unique ID for each resume
    )
    
    # Save resume details to a JSON file
    save_resume_metadata(resume_details)
    
    return resume_details

# Save resume metadata to JSON
def save_resume_metadata(metadata):
    resumes = load_resumes()
    resumes.append(metadata)
    save_resumes(resumes)

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
        if isinstance(score, list):
            score = score[0]  # or handle it accordingly, e.g., average the list

        similarity_score = round((1 - score) * 100, 2)  # Convert distance to similarity score (percentage)

        
        # Add resume to the ranked list with detailed information
        ranked_resumes.append({
            "name": metadata.get("name", ""),
            "email": metadata.get("email", ""),
            "phone": metadata.get("phone", ""),
            "dob": metadata.get("dob", ""),
            "skills": metadata.get("skills", ""),
            "score": similarity_score  # Return the calculated similarity score
        })
    
    # Return the ranked resumes list
    return ranked_resumes

# @app.post("/generate-job-description")
# async def generate_job_description(request: JobDescriptionRequest):
#     prompt = f"Create a detailed job description for the following skills: {', '.join(request.skills)}"
#     job_description = generate_text_with_bedrock(prompt)
#     return {"job_description": job_description}


# @app.post("/generate-interview-questions")
# async def generate_interview_questions(request: InterviewQuestionsRequest):
#     prompt = f"Based on the following {request.input_type}, generate 5 interview questions:\n\n{request.input_text}"
#     interview_questions = generate_text_with_bedrock(prompt)
#     return {"interview_questions": interview_questions}


# @app.post("/upload-resume")
# async def upload_resume(file: UploadFile = File(...)):
#     try:
#         file_path = f"./{file.filename}"
#         with open(file_path, "wb") as f:
#             content = await file.read()
#             f.write(content)

#         # Upload the file to S3 and process it
#         metadata = upload_resume_to_s3_and_process(file_path, file.filename)
#         os.remove(file_path)  # Clean up local file after processing

#         return {"message": "File uploaded and processed successfully", "metadata": metadata}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error uploading or processing the resume: {str(e)}")

# # FastAPI endpoint to rank resumes based on job description query
# @app.post("/rank-resumes")
# async def rank_resumes(query: str):
#     ranked_resumes = rank_resumes_based_on_query(query)
#     return {"ranked_resumes": ranked_resumes}


# # Add resume to ChromaDB with metadata
# @app.post("/add-resume-to-chromadb")
# async def add_resume_to_chromadb(resume_text: str, metadata: ResumeMetadata, file_name: str):
#     try:
#         resume_embedding = generate_resume_embedding(resume_text)
#         collection.add(
#             documents=[resume_text],
#             metadatas=[metadata.dict()],  # This converts the metadata to a dictionary format
#             embeddings=[resume_embedding],
#             ids=[file_name]
#         )
#         return {"message": "Resume added to ChromaDB"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error adding resume to ChromaDB: {str(e)}")



# # Rank resumes based on query
# @app.post("/rank-resumes")
# async def rank_resumes_based_on_query(query: str, ranking_type: str = "skills"):
#     try:
#         query_embedding = generate_resume_embedding(query)

#         results = collection.query(
#             query_embeddings=[query_embedding],
#             n_results=5
#         )

#         ranked_resumes = []
#         for result in results['documents']:
#             resume_metadata = result['metadata']
#             resume_score = cosine_similarity([query_embedding], [result['embedding']])[0][0]
#             ranked_resumes.append({
#                 'file_name': result['id'],
#                 'skills': resume_metadata.get('skills', ''),
#                 'name': resume_metadata.get('name', ''),
#                 'dob': resume_metadata.get('dob', ''),
#                 'email': resume_metadata.get('email', ''),
#                 'phone': resume_metadata.get('phone', ''),
#                 'score': round(resume_score * 100, 2)  # Convert to percentage
#             })

#         return {"ranked_resumes": ranked_resumes}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error ranking resumes: {str(e)}")


# # Get all resumes from ChromaDB
# @app.get("/get-all-resumes")
# async def get_all_resumes():
#     try:
#         results = collection.query(n_results=5)

#         resumes = []
#         for result in results['documents']:
#             resume_metadata = result['metadata']
#             resumes.append({
#                 'file_name': result['id'],
#                 'skills': resume_metadata.get('skills', ''),
#                 'name': resume_metadata.get('name', ''),
#                 'dob': resume_metadata.get('dob', ''),
#                 'email': resume_metadata.get('email', ''),
#                 'phone': resume_metadata.get('phone', ''),
#             })

#         return {"resumes": resumes}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error fetching resumes: {str(e)}")


# # Root endpoint for checking if the API is working
# @app.get("/")
# async def read_root():
#     return {"message": "FastAPI is running. Use /docs for API documentation."}


