from typing import Dict, Any, List
import pandas as pd
import os
from .base_agent import BaseAgent
from utils.helpers import DocumentProcessor, TextProcessor, DataValidator
from config.settings import Config

class ScraperAgent(BaseAgent):
    """Agent responsible for scraping and extracting information from documents"""
    
    def __init__(self, config: Config = None):
        super().__init__('scraper', config)
        self.document_processor = DocumentProcessor()
        self.text_processor = TextProcessor()
        self.data_validator = DataValidator()
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for scraping"""
        required_keys = ['job_description_path', 'candidates_csv_path']
        
        for key in required_keys:
            if key not in input_data:
                self.log_action(f'Missing required key: {key}')
                return False
            
            file_path = input_data[key]
            if not os.path.exists(file_path):
                self.log_action(f'File not found: {file_path}')
                return False
        
        return True
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process documents and extract information"""
        try:
            self.update_status('processing')
            
            if not self.validate_input(input_data):
                return self.handle_error(ValueError("Invalid input data"), "Input validation failed")
            
            # Extract job description
            jd_path = input_data['job_description_path']
            candidates_path = input_data['candidates_csv_path']
            
            self.log_action('Extracting job description', {'file_path': jd_path})
            job_description = self._extract_job_description(jd_path)
            
            self.log_action('Processing candidates data', {'file_path': candidates_path})
            candidates_data = self._extract_candidates_data(candidates_path)
            
            # Validate extracted data
            jd_validation = self.data_validator.validate_job_description(job_description['text'])
            candidates_validation = self.data_validator.validate_candidate_data(candidates_data['dataframe'])
            
            results = {
                'job_description': job_description,
                'candidates': candidates_data,
                'validation': {
                    'job_description': jd_validation,
                    'candidates': candidates_validation
                },
                'metadata': {
                    'total_candidates': len(candidates_data['dataframe']),
                    'processing_timestamp': input_data.get('timestamp'),
                    'files_processed': [jd_path, candidates_path]
                }
            }
            
            self.results = results
            self.update_status('completed')
            
            return {
                'success': True,
                'results': results,
                'agent_info': self.get_info()
            }
            
        except Exception as e:
            return self.handle_error(e, "Document processing failed")
    
    def _extract_job_description(self, pdf_path: str) -> Dict[str, Any]:
        """Extract and analyze job description from PDF"""
        try:
            # Extract text from PDF
            raw_text = self.document_processor.extract_text_from_pdf(pdf_path)
            
            if not raw_text:
                raise ValueError("No text extracted from PDF")
            
            # Clean and process text
            cleaned_text = self.text_processor.clean_text(raw_text)
            
            # Extract structured information
            skills = self.text_processor.extract_skills_from_text(cleaned_text)
            contact_info = self.text_processor.extract_contact_info(cleaned_text)
            
            # Extract additional job details
            job_details = self._extract_job_details(cleaned_text)
            
            return {
                'text': cleaned_text,
                'raw_text': raw_text,
                'skills_required': skills,
                'contact_info': contact_info,
                'job_details': job_details,
                'word_count': len(cleaned_text.split()),
                'file_path': pdf_path
            }
            
        except Exception as e:
            self.log_action(f'Error extracting job description: {str(e)}')
            raise
    
    def _extract_candidates_data(self, csv_path: str) -> Dict[str, Any]:
        """Extract and process candidates data from CSV"""
        try:
            # Read CSV file
            df = self.document_processor.process_csv_file(csv_path)
            
            if df.empty:
                raise ValueError("No data found in CSV file")
            
            # Standardize column names
            df = self._standardize_columns(df)
            print('DEBUG: Columns after standardization:', df.columns.tolist())
            print('DEBUG: First few rows:', df.head().to_dict())
            
            # Process each candidate
            processed_candidates = []
            for index, row in df.iterrows():
                candidate = self._process_candidate_row(row, index)
                processed_candidates.append(candidate)
            
            return {
                'dataframe': df,
                'processed_candidates': processed_candidates,
                'total_count': len(df),
                'file_path': csv_path
            }
            
        except Exception as e:
            self.log_action(f'Error processing candidates CSV: {str(e)}')
            raise
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to expected format"""
        column_mapping = {
            'name': ['name', 'full_name', 'candidate_name', 'applicant_name', 'applicant name'],
            'email': ['email', 'email_address', 'contact_email', 'e_mail'],
            'phone': ['phone', 'phone_number', 'contact_number', 'mobile', 'phone number'],
            'skills': ['skills', 'technical_skills', 'expertise', 'competencies'],
            'experience': ['experience', 'work_experience', 'years_experience', 'exp'],
            'education': ['education', 'qualification', 'degree', 'academic_background', 'academic background'],
            'location': ['location', 'city', 'address', 'current_location']
        }
        # Clean up column names for matching (strip, lower, replace underscores with spaces)
        def normalize(col):
            return col.strip().lower().replace('_', ' ')
        cleaned_columns = {col: normalize(col) for col in df.columns}
        rename_dict = {}
        for standard_col, possible_names in column_mapping.items():
            normalized_possible = [name.lower().replace('_', ' ') for name in possible_names]
            for col, cleaned_col in cleaned_columns.items():
                if cleaned_col in normalized_possible:
                    rename_dict[col] = standard_col
                    break
        return df.rename(columns=rename_dict)
    
    def _process_candidate_row(self, row: pd.Series, index: int) -> Dict[str, Any]:
        """Process individual candidate row"""
        # Fix: handle missing/empty/NaN name gracefully
        name = row.get('name', '')
        if pd.isna(name) or str(name).strip().lower() in ['nan', 'none', 'unknown', '']:
            name = ''
        else:
            name = str(name).strip()
        candidate = {
            'index': index,
            'name': name,
            'email': str(row.get('email', '')),
            'phone': str(row.get('phone', '')),
            'raw_skills': str(row.get('skills', '')),
            'experience': str(row.get('experience', '')),
            'education': str(row.get('education', '')),
            'location': str(row.get('location', ''))
        }
        
        # Extract skills from text
        skills_text = f"{candidate['raw_skills']} {candidate['experience']} {candidate['education']}"
        extracted_skills = self.text_processor.extract_skills_from_text(skills_text)
        candidate['extracted_skills'] = extracted_skills
        
        # Create a profile summary
        candidate['profile_summary'] = self._create_profile_summary(candidate)
        
        return candidate
    
    def _create_profile_summary(self, candidate: Dict[str, Any]) -> str:
        """Create a summary profile for the candidate"""
        summary_parts = []
        
        if candidate['name'] and candidate['name'] != 'Unknown':
            summary_parts.append(f"Name: {candidate['name']}")
        
        if candidate['experience']:
            summary_parts.append(f"Experience: {candidate['experience']}")
        
        if candidate['education']:
            summary_parts.append(f"Education: {candidate['education']}")
        
        if candidate['extracted_skills']:
            summary_parts.append(f"Skills: {', '.join(candidate['extracted_skills'])}")
        
        if candidate['location']:
            summary_parts.append(f"Location: {candidate['location']}")
        
        return ". ".join(summary_parts)
    
    def _extract_job_details(self, text: str) -> Dict[str, Any]:
        """Extract specific job details from text"""
        job_details = {
            'title': '',
            'company': '',
            'location': '',
            'salary_range': '',
            'job_type': '',
            'experience_level': ''
        }
        
        # Simple pattern matching for job details
        import re
        
        # Extract job title (usually at the beginning)
        title_patterns = [
            r'(?i)position[:\s]+([^\n]+)',
            r'(?i)role[:\s]+([^\n]+)',
            r'(?i)job\s+title[:\s]+([^\n]+)',
            r'(?i)we\s+are\s+looking\s+for\s+a\s+([^\n]+)'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text)
            if match:
                job_details['title'] = match.group(1).strip()
                break
        
        # Extract company name
        company_patterns = [
            r'(?i)company[:\s]+([^\n]+)',
            r'(?i)organization[:\s]+([^\n]+)',
            r'(?i)about\s+([A-Z][a-zA-Z\s]+)(?=\s+is\s+)'
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text)
            if match:
                job_details['company'] = match.group(1).strip()
                break
        
        # Extract experience level
        exp_patterns = [
            r'(?i)(\d+\+?\s*years?\s*of?\s*experience)',
            r'(?i)(junior|senior|mid-level|entry-level|experienced)',
            r'(?i)experience[:\s]+([^\n]+)'
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, text)
            if match:
                job_details['experience_level'] = match.group(1).strip()
                break
        
        return job_details
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """Get summary of extraction results"""
        if not self.results:
            return {'status': 'No processing completed yet'}
        
        return {
            'status': self.status,
            'job_description_extracted': bool(self.results.get('job_description', {}).get('text')),
            'candidates_processed': self.results.get('metadata', {}).get('total_candidates', 0),
            'skills_found_in_jd': len(self.results.get('job_description', {}).get('skills_required', [])),
            'validation_results': self.results.get('validation', {})
        }
