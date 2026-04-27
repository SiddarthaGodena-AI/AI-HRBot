# AI-HRBot
HRBot - AI Powered Resume Screening & Recruitment Assistant

An end-to-end AI-driven HR automation backend built with FastAPI, leveraging AWS Bedrock, ChromaDB, and Sentence Transformers to streamline hiring workflows.

This system helps recruiters:

Generate job descriptions
Create interview questions
Upload & parse resumes
Extract candidate details
Rank candidates intelligently
🧠 Key Features
📄 1. Resume Processing
Supports multiple formats:
PDF (PyMuPDF)
DOC/DOCX (python-docx)
Images (Tesseract OCR)
Extracts:
Name, Email, Phone, DOB
Skills
🤖 2. AI-Powered Capabilities (AWS Bedrock)
Generate Job Descriptions from skills
Generate Interview Questions
Extract structured data from resumes
🔍 3. Semantic Resume Ranking
Uses Sentence Transformers (all-MiniLM-L6-v2)
Stores embeddings in ChromaDB
Ranks resumes using cosine similarity
Returns a match score (%)
☁️ 4. Cloud Integration (AWS)
Resume storage using S3
AI inference using Bedrock Runtime
📦 5. Persistent Vector Database
Uses ChromaDB (Persistent Client)
Efficient similarity search for resumes
🛠️ Tech Stack
Backend: FastAPI
LLM: AWS Bedrock (AI21 Jurassic)
Embeddings: Sentence Transformers
Vector DB: ChromaDB
OCR: Tesseract
File Parsing: PyMuPDF, python-docx
Cloud: AWS S3
📌 API Endpoints
🔹 Generate Job Description
POST /generate-job-description

Input:

{
  "skills": ["Python", "Machine Learning"]
}
🔹 Generate Interview Questions
POST /generate-interview-questions
🔹 Upload Resume
POST /upload-resume
Upload file (PDF/DOCX/Image)
Extracts metadata
Stores embeddings in ChromaDB
🔹 Rank Resumes
POST /rank-resumes

Input:

{
  "query": "Looking for a Python ML Engineer"
}
🔹 Get All Resumes
GET /get-all-resumes
⚙️ Setup Instructions
1️⃣ Clone Repo
git clone https://github.com/your-username/hrbot.git
cd hrbot
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Install Tesseract OCR
Download: https://github.com/tesseract-ocr/tesseract
Update path in code:
pytesseract.pytesseract.tesseract_cmd = "YOUR_PATH"
4️⃣ Configure AWS

Set environment variables:

export AWS_ACCESS_KEY_ID=YOUR_KEY
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET
export AWS_DEFAULT_REGION=us-east-1
5️⃣ Run Server
uvicorn main:app --reload
6️⃣ Access API Docs
http://localhost:8000/docs
📊 How It Works
Upload Resume
Extract Text (OCR / PDF / DOCX)
Generate Metadata (Bedrock)
Create Embeddings
Store in ChromaDB
Query with Job Description
Rank using Similarity Score
