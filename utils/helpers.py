import pandas as pd
import PyPDF2
import fitz  # PyMuPDF
import re
import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

class DocumentProcessor:
    """Utility class for processing various document formats"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from PDF file using PyMuPDF for better accuracy"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    return text.strip()
            except Exception as e2:
                print(f"Fallback PDF extraction also failed: {e2}")
                return ""
    
    @staticmethod
    def process_csv_file(csv_path: str) -> pd.DataFrame:
        """Process CSV file and return DataFrame"""
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            print(f"Error processing CSV file: {e}")
            return pd.DataFrame()

class TextProcessor:
    """Utility class for text processing and cleaning"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep alphanumeric and common punctuation
        text = re.sub(r'[^\w\s\-.,;:()[\]{}]', '', text)
        return text.strip()
    
    @staticmethod
    def extract_skills_from_text(text: str) -> List[str]:
        """Extract skills from text using pattern matching"""
        # Common skill patterns
        skill_patterns = [
            r'\b(?:Python|Java|JavaScript|React|Angular|Vue|Node\.js|Django|Flask)\b',
            r'\b(?:SQL|MySQL|PostgreSQL|MongoDB|Redis|ElasticSearch)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git)\b',
            r'\b(?:Machine Learning|Deep Learning|AI|NLP|Computer Vision)\b',
            r'\b(?:Agile|Scrum|DevOps|CI/CD|TDD|BDD)\b',
            r'\b(?:HTML|CSS|Bootstrap|Tailwind|SASS|LESS)\b',
            r'\b(?:REST|GraphQL|API|Microservices|WebSocket)\b'
        ]
        
        skills = []
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend(matches)
        
        return list(set(skills))  # Remove duplicates
    
    @staticmethod
    def extract_contact_info(text: str) -> Dict[str, str]:
        """Extract contact information from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        
        emails = re.findall(email_pattern, text)
        phones = re.findall(phone_pattern, text)
        
        return {
            'email': emails[0] if emails else '',
            'phone': phones[0] if phones else ''
        }

class VectorUtils:
    """Utility class for vector operations"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        return self.model.encode([text])[0]
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        return self.model.encode(texts)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

class EmailUtils:
    """Utility class for email operations"""
    
    @staticmethod
    def generate_email_template(candidate_name: str, position: str, company: str, 
                              match_score: float, key_skills: List[str]) -> str:
        """Generate personalized email template"""
        template = f"""
Subject: Exciting Opportunity at {company} - {position}

Dear {candidate_name},

I hope this email finds you well. I am reaching out regarding an exciting {position} opportunity at {company}.

Based on your profile, I believe you would be an excellent fit for this role. Your skills in {', '.join(key_skills[:3])} particularly align well with our requirements (Match Score: {match_score:.0%}).

Key highlights of this opportunity:
- Work with cutting-edge technology
- Collaborative team environment
- Competitive compensation package
- Growth opportunities

I would love to discuss this opportunity with you in more detail. Are you available for a brief call this week?

Best regards,
Recruitment Team
{company}

---
This is an automated message from our AI-powered recruitment system.
"""
        return template.strip()
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

class LoggingUtils:
    """Utility class for logging operations"""
    
    @staticmethod
    def log_agent_action(agent_name: str, action: str, details: Dict[str, Any]) -> None:
        """Log agent actions for debugging and monitoring"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'agent': agent_name,
            'action': action,
            'details': details
        }
        print(f"[{timestamp}] {agent_name}: {action}")
        print(f"Details: {json.dumps(details, indent=2)}")
    
    @staticmethod
    def save_results(results: Dict[str, Any], filename: str) -> None:
        """Save results to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")

class DataValidator:
    """Utility class for data validation"""
    
    @staticmethod
    def validate_candidate_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate candidate data structure"""
        required_columns = ['name', 'email', 'skills', 'experience']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        validation_result = {
            'is_valid': len(missing_columns) == 0,
            'missing_columns': missing_columns,
            'total_records': len(df),
            'valid_emails': df['email'].apply(EmailUtils.validate_email).sum() if 'email' in df.columns else 0
        }
        
        return validation_result
    
    @staticmethod
    def validate_job_description(text: str) -> Dict[str, Any]:
        """Validate job description content"""
        validation_result = {
            'is_valid': len(text.strip()) > 100,
            'word_count': len(text.split()),
            'has_skills': len(TextProcessor.extract_skills_from_text(text)) > 0,
            'has_contact': bool(TextProcessor.extract_contact_info(text)['email'])
        }
        
        return validation_result
