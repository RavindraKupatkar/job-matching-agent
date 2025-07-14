import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    # API Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp-free")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Email Configuration
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    
    # Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")
    
    # Vector Database Configuration
    VECTOR_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2
    INDEX_NAME = "recruitment-candidates"
    
    # Matching Configuration
    TOP_K_MATCHES = 10
    SIMILARITY_THRESHOLD = 0.7
    
    # File Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    UPLOAD_DIR = DATA_DIR / "uploads"
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)

# Agent Configuration
AGENT_ROLES = {
    "scraper": {
        "name": "Document Scraper Agent",
        "description": "Extracts and processes information from PDF and CSV files",
        "capabilities": ["pdf_extraction", "csv_processing", "data_cleaning"]
    },
    "matcher": {
        "name": "Candidate Matching Agent", 
        "description": "Matches candidates with job requirements using vector similarity",
        "capabilities": ["embedding_generation", "similarity_search", "ranking"]
    },
    "analyzer": {
        "name": "Skills Analysis Agent",
        "description": "Analyzes job requirements and candidate skills",
        "capabilities": ["skill_extraction", "requirement_analysis", "competency_mapping"]
    },
    "emailer": {
        "name": "Email Communication Agent",
        "description": "Handles email composition and sending to matched candidates",
        "capabilities": ["email_composition", "personalization", "email_sending"]
    },
    "coordinator": {
        "name": "Workflow Coordinator Agent",
        "description": "Orchestrates the entire recruitment workflow",
        "capabilities": ["workflow_management", "agent_coordination", "result_aggregation"]
    }
}
