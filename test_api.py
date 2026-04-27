import requests

BASE_URL = "http://127.0.0.1:8000"

# Test the root endpoint
def test_root():
    response = requests.get(f"{BASE_URL}/")
    print("Root Endpoint Response:", response.json())

# Test job description generation
def test_generate_job_description():
    data = {"skills": "Python, Machine Learning, Data Analysis"}
    response = requests.post(f"{BASE_URL}/generate-job-description", json=data)
    print("Job Description Response:", response.json())

# Test interview questions generation
def test_generate_interview_questions():
    data = {"input_text": "Software Engineer role requiring Python expertise", "input_type": "Job Description"}
    response = requests.post(f"{BASE_URL}/generate-interview-questions", json=data)
    print("Interview Questions Response:", response.json())

# Test resume upload
def test_upload_resume():
    file_path = "sample_resume.pdf"  # Path to a sample resume file
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/upload-resume", files=files)
    print("Upload Resume Response:", response.json())

# Test adding resume to ChromaDB
def test_add_resume_to_chromadb():
    resume_text = "John Doe is a Python developer with 5 years of experience in Machine Learning."
    metadata = {
        "skills": ["Python", "Machine Learning"],
        "name": "John Doe",
        "dob": "1990-01-01",
        "email": "johndoe@example.com",
        "phone": "123-456-7890"
    }
    data = {"resume_text": resume_text, "metadata": metadata, "file_name": "john_doe_resume"}
    response = requests.post(f"{BASE_URL}/add-resume-to-chromadb", json=data)
    print("Add Resume to ChromaDB Response:", response.json())

# Test ranking resumes
def test_rank_resumes():
    data = {"query": "Python, Machine Learning", "ranking_type": "skills"}
    response = requests.post(f"{BASE_URL}/rank-resumes", json=data)
    print("Rank Resumes Response:", response.json())

# Test fetching all resumes
def test_get_all_resumes():
    response = requests.get(f"{BASE_URL}/get-all-resumes")
    print("Get All Resumes Response:", response.json())

# Run all tests
if __name__ == "__main__":
    print("Testing API Endpoints...\n")
    test_root()
    test_generate_job_description()
    test_generate_interview_questions()
    test_upload_resume()
    test_add_resume_to_chromadb()
    test_rank_resumes()
    test_get_all_resumes()
