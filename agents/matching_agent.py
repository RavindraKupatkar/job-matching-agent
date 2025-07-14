from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pinecone
from .base_agent import BaseAgent
from utils.helpers import VectorUtils
from config.settings import Config

class MatchingAgent(BaseAgent):
    """Agent responsible for matching candidates with job requirements using vector similarity"""
    
    def __init__(self, config: Config = None):
        super().__init__('matcher', config)
        self.vector_utils = VectorUtils(self.config.EMBEDDING_MODEL)
        self.pinecone_client = None
        self.index = None
        self._initialize_vector_db()
        
    def _initialize_vector_db(self):
        """Initialize Pinecone vector database"""
        try:
            if self.config.PINECONE_API_KEY:
                pinecone.init(
                    api_key=self.config.PINECONE_API_KEY,
                    environment=self.config.PINECONE_ENVIRONMENT
                )
                self.pinecone_client = pinecone
                self.log_action('Pinecone initialized successfully')
            else:
                self.log_action('No Pinecone API key provided, using local similarity matching')
        except Exception as e:
            self.log_action(f'Error initializing Pinecone: {str(e)}')
            self.pinecone_client = None
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for matching"""
        required_keys = ['job_description', 'candidates']
        
        for key in required_keys:
            if key not in input_data:
                self.log_action(f'Missing required key: {key}')
                return False
        
        # Validate job description
        job_desc = input_data['job_description']
        if not job_desc.get('text'):
            self.log_action('Job description text is empty')
            return False
        
        # Validate candidates
        candidates = input_data['candidates']
        if not candidates.get('processed_candidates'):
            self.log_action('No processed candidates found')
            return False
        
        return True
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process job description and candidates to find matches"""
        try:
            self.update_status('processing')
            
            if not self.validate_input(input_data):
                return self.handle_error(ValueError("Invalid input data"), "Input validation failed")
            
            job_description = input_data['job_description']
            candidates = input_data['candidates']['processed_candidates']
            
            # Generate embeddings
            self.log_action('Generating embeddings for job description')
            job_embedding = self._generate_job_embedding(job_description)
            
            self.log_action('Generating embeddings for candidates', {'count': len(candidates)})
            candidate_embeddings = self._generate_candidate_embeddings(candidates)
            
            # Perform matching
            self.log_action('Performing similarity matching')
            matches = self._perform_matching(job_embedding, candidate_embeddings, candidates)
            
            # Rank and filter matches
            ranked_matches = self._rank_matches(matches)
            top_matches = self._filter_top_matches(ranked_matches)
            
            # Generate detailed analysis
            match_analysis = self._analyze_matches(top_matches, job_description)
            
            results = {
                'matches': top_matches,
                'match_analysis': match_analysis,
                'job_embedding': job_embedding.tolist(),
                'total_candidates_evaluated': len(candidates),
                'matches_found': len(top_matches),
                'matching_threshold': self.config.SIMILARITY_THRESHOLD
            }
            
            self.results = results
            self.update_status('completed')
            
            return {
                'success': True,
                'results': results,
                'agent_info': self.get_info()
            }
            
        except Exception as e:
            return self.handle_error(e, "Matching process failed")
    
    def _generate_job_embedding(self, job_description: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for job description"""
        # Combine different aspects of job description
        job_text_parts = [
            job_description.get('text', ''),
            f"Required skills: {', '.join(job_description.get('skills_required', []))}",
            f"Job details: {job_description.get('job_details', {}).get('title', '')}"
        ]
        
        job_text = ' '.join(filter(None, job_text_parts))
        return self.vector_utils.generate_embedding(job_text)
    
    def _generate_candidate_embeddings(self, candidates: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Generate embeddings for all candidates"""
        candidate_texts = []
        
        for candidate in candidates:
            # Create comprehensive candidate profile text
            profile_parts = [
                candidate.get('profile_summary', ''),
                f"Skills: {', '.join(candidate.get('extracted_skills', []))}",
                candidate.get('experience', ''),
                candidate.get('education', '')
            ]
            
            candidate_text = ' '.join(filter(None, profile_parts))
            candidate_texts.append(candidate_text)
        
        # Generate embeddings in batch for efficiency
        embeddings = self.vector_utils.generate_embeddings_batch(candidate_texts)
        return embeddings
    
    def _perform_matching(self, job_embedding: np.ndarray, 
                         candidate_embeddings: List[np.ndarray], 
                         candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform similarity matching between job and candidates"""
        matches = []
        
        for i, candidate_embedding in enumerate(candidate_embeddings):
            # Calculate similarity
            similarity = self.vector_utils.calculate_similarity(job_embedding, candidate_embedding)
            
            # Create match object
            match = {
                'candidate': candidates[i],
                'similarity_score': float(similarity),
                'candidate_embedding': candidate_embedding.tolist(),
                'match_id': f"match_{i}_{int(similarity * 1000)}"
            }
            
            matches.append(match)
        
        return matches
    
    def _rank_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank matches by similarity score and other factors"""
        # Sort by similarity score (descending)
        ranked_matches = sorted(matches, key=lambda x: x['similarity_score'], reverse=True)
        
        # Add ranking information
        for i, match in enumerate(ranked_matches):
            match['rank'] = i + 1
            match['percentile'] = ((len(ranked_matches) - i) / len(ranked_matches)) * 100
        
        return ranked_matches
    
    def _filter_top_matches(self, ranked_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter top matches based on threshold and limit"""
        # Filter by similarity threshold
        filtered_matches = [
            match for match in ranked_matches 
            if match['similarity_score'] >= self.config.SIMILARITY_THRESHOLD
        ]
        
        # Limit to top K matches
        top_matches = filtered_matches[:self.config.TOP_K_MATCHES]
        
        return top_matches
    
    def _analyze_matches(self, matches: List[Dict[str, Any]], 
                        job_description: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze matching results and provide insights"""
        if not matches:
            return {
                'summary': 'No matches found above threshold',
                'insights': [],
                'recommendations': ['Consider lowering similarity threshold', 'Review job requirements']
            }
        
        # Calculate statistics
        scores = [match['similarity_score'] for match in matches]
        
        analysis = {
            'summary': f'Found {len(matches)} qualified candidates',
            'score_statistics': {
                'max_score': max(scores),
                'min_score': min(scores),
                'avg_score': np.mean(scores),
                'std_score': np.std(scores)
            },
            'insights': [],
            'recommendations': []
        }
        
        # Generate insights
        required_skills = job_description.get('skills_required', [])
        if required_skills:
            skill_coverage = self._analyze_skill_coverage(matches, required_skills)
            analysis['skill_coverage'] = skill_coverage
            analysis['insights'].append(f"Top skills in matches: {', '.join(skill_coverage.get('top_skills', [])[:3])}")
        
        # Add recommendations
        if len(matches) < 3:
            analysis['recommendations'].append('Consider expanding search criteria')
        
        if max(scores) < 0.8:
            analysis['recommendations'].append('Consider reviewing job requirements for better matches')
        
        return analysis
    
    def _analyze_skill_coverage(self, matches: List[Dict[str, Any]], 
                               required_skills: List[str]) -> Dict[str, Any]:
        """Analyze skill coverage in matches"""
        skill_counts = {}
        total_matches = len(matches)
        
        for match in matches:
            candidate_skills = match['candidate'].get('extracted_skills', [])
            for skill in candidate_skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Calculate coverage percentages
        skill_coverage = {
            skill: (count / total_matches) * 100 
            for skill, count in skill_counts.items()
        }
        
        # Find top skills
        top_skills = sorted(skill_coverage.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Check coverage of required skills
        required_skill_coverage = {
            skill: skill_coverage.get(skill, 0) 
            for skill in required_skills
        }
        
        return {
            'top_skills': [skill for skill, _ in top_skills],
            'skill_coverage': skill_coverage,
            'required_skill_coverage': required_skill_coverage,
            'total_unique_skills': len(skill_counts)
        }
    
    def store_embeddings_in_pinecone(self, matches: List[Dict[str, Any]], 
                                   job_id: str) -> bool:
        """Store embeddings in Pinecone for future use"""
        if not self.pinecone_client:
            self.log_action('Pinecone not available, skipping storage')
            return False
        
        try:
            # Create or connect to index
            if self.config.INDEX_NAME not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.config.INDEX_NAME,
                    dimension=self.config.VECTOR_DIMENSION,
                    metric='cosine'
                )
            
            index = pinecone.Index(self.config.INDEX_NAME)
            
            # Prepare vectors for upsert
            vectors = []
            for match in matches:
                vector_id = f"{job_id}_{match['match_id']}"
                metadata = {
                    'job_id': job_id,
                    'candidate_name': match['candidate']['name'],
                    'candidate_email': match['candidate']['email'],
                    'similarity_score': match['similarity_score'],
                    'rank': match['rank']
                }
                
                vectors.append({
                    'id': vector_id,
                    'values': match['candidate_embedding'],
                    'metadata': metadata
                })
            
            # Upsert vectors
            index.upsert(vectors)
            self.log_action(f'Stored {len(vectors)} embeddings in Pinecone')
            return True
            
        except Exception as e:
            self.log_action(f'Error storing embeddings in Pinecone: {str(e)}')
            return False
    
    def get_matching_summary(self) -> Dict[str, Any]:
        """Get summary of matching results"""
        if not self.results:
            return {'status': 'No matching completed yet'}
        
        return {
            'status': self.status,
            'total_candidates_evaluated': self.results.get('total_candidates_evaluated', 0),
            'matches_found': self.results.get('matches_found', 0),
            'matching_threshold': self.results.get('matching_threshold', 0),
            'top_match_score': self.results.get('matches', [{}])[0].get('similarity_score', 0) if self.results.get('matches') else 0
        }
