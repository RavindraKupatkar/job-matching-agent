from typing import Dict, Any, List
from datetime import datetime
from .base_agent import BaseAgent, AgentRegistry
from .scraper_agent import ScraperAgent
from .matching_agent import MatchingAgent
from .emailer_agent import EmailerAgent
from config.settings import Config
from utils.helpers import LoggingUtils

class CoordinatorAgent(BaseAgent):
    """Agent responsible for orchestrating the entire recruitment workflow"""
    
    def __init__(self, config: Config = None):
        super().__init__('coordinator', config)
        self.registry = AgentRegistry()
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize all specialized agents"""
        try:
            # Create and register agents
            scraper = ScraperAgent(self.config)
            matcher = MatchingAgent(self.config)
            emailer = EmailerAgent(self.config)
            
            self.registry.register_agent('scraper', scraper)
            self.registry.register_agent('matcher', matcher)
            self.registry.register_agent('emailer', emailer)
            
            self.log_action('All agents initialized successfully')
            
        except Exception as e:
            self.log_action(f'Error initializing agents: {str(e)}')
            raise
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for the entire workflow"""
        required_keys = ['job_description_path', 'candidates_csv_path']
        
        for key in required_keys:
            if key not in input_data:
                self.log_action(f'Missing required key: {key}')
                return False
        
        return True
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete recruitment workflow"""
        try:
            self.update_status('processing')
            workflow_start_time = datetime.now()
            
            if not self.validate_input(input_data):
                return self.handle_error(ValueError("Invalid input data"), "Input validation failed")
            
            # Add timestamp to input data
            input_data['timestamp'] = workflow_start_time.isoformat()
            
            # Step 1: Document Scraping
            self.log_action('Starting document scraping phase')
            scraper_result = self._execute_scraping(input_data)
            
            if not scraper_result['success']:
                return scraper_result
            
            # Step 2: Candidate Matching
            self.log_action('Starting candidate matching phase')
            matching_result = self._execute_matching(scraper_result['results'])
            
            if not matching_result['success']:
                return matching_result
            
            # Step 3: Email Sending (if requested)
            email_result = None
            if input_data.get('send_emails', False):
                self.log_action('Starting email sending phase')
                email_result = self._execute_email_sending(
                    matching_result['results'], 
                    scraper_result['results']['job_description']
                )
            
            # Compile final results
            workflow_end_time = datetime.now()
            execution_time = (workflow_end_time - workflow_start_time).total_seconds()
            
            final_results = {
                'workflow_summary': {
                    'execution_time_seconds': execution_time,
                    'start_time': workflow_start_time.isoformat(),
                    'end_time': workflow_end_time.isoformat(),
                    'total_candidates_processed': scraper_result['results']['metadata']['total_candidates'],
                    'matches_found': matching_result['results']['matches_found'],
                    'emails_sent': email_result['results']['emails_sent'] if email_result else 0
                },
                'scraping_results': scraper_result['results'],
                'matching_results': matching_result['results'],
                'email_results': email_result['results'] if email_result else None,
                'agent_status': self.registry.get_system_status()
            }
            
            self.results = final_results
            self.update_status('completed')
            
            return {
                'success': True,
                'results': final_results,
                'agent_info': self.get_info()
            }
            
        except Exception as e:
            return self.handle_error(e, "Workflow execution failed")
    
    def _execute_scraping(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the document scraping phase"""
        try:
            scraper = self.registry.get_agent('scraper')
            result = scraper.process(input_data)
            
            self.log_action('Document scraping completed', {
                'success': result['success'],
                'candidates_found': result['results']['metadata']['total_candidates'] if result['success'] else 0
            })
            
            return result
            
        except Exception as e:
            return self.handle_error(e, "Scraping phase failed")
    
    def _execute_matching(self, scraper_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the candidate matching phase"""
        try:
            matcher = self.registry.get_agent('matcher')
            
            # Prepare matching input
            matching_input = {
                'job_description': scraper_results['job_description'],
                'candidates': scraper_results['candidates']
            }
            
            result = matcher.process(matching_input)
            
            self.log_action('Candidate matching completed', {
                'success': result['success'],
                'matches_found': result['results']['matches_found'] if result['success'] else 0
            })
            
            return result
            
        except Exception as e:
            return self.handle_error(e, "Matching phase failed")
    
    def _execute_email_sending(self, matching_results: Dict[str, Any], 
                              job_description: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the email sending phase"""
        try:
            emailer = self.registry.get_agent('emailer')
            
            # Prepare email input
            email_input = {
                'matches': matching_results['matches'],
                'job_details': job_description['job_details']
            }
            
            result = emailer.process(email_input)
            
            self.log_action('Email sending completed', {
                'success': result['success'],
                'emails_sent': result['results']['emails_sent'] if result['success'] else 0
            })
            
            return result
            
        except Exception as e:
            return self.handle_error(e, "Email sending phase failed")
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            'coordinator_status': self.status,
            'system_status': self.registry.get_system_status(),
            'last_execution_results': self.results
        }
    
    def reset_workflow(self) -> None:
        """Reset the entire workflow"""
        self.reset()
        
        # Reset all agents
        for agent in self.registry.get_all_agents().values():
            agent.reset()
        
        self.log_action('Workflow reset completed')
    
    def get_agent_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all agents"""
        performance = {}
        
        for agent_id, agent in self.registry.get_all_agents().items():
            performance[agent_id] = {
                'status': agent.status,
                'last_action': agent.last_action,
                'has_results': bool(agent.results)
            }
        
        return performance
    
    def execute_custom_workflow(self, workflow_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a custom workflow with specific steps"""
        try:
            self.update_status('processing_custom')
            results = {}
            
            for step in workflow_steps:
                agent_id = step['agent']
                input_data = step['input']
                
                agent = self.registry.get_agent(agent_id)
                if not agent:
                    raise ValueError(f"Agent '{agent_id}' not found")
                
                step_result = agent.process(input_data)
                results[agent_id] = step_result
                
                # If any step fails, stop the workflow
                if not step_result['success']:
                    break
            
            self.update_status('completed')
            return {
                'success': True,
                'results': results
            }
            
        except Exception as e:
            return self.handle_error(e, "Custom workflow execution failed")
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of the recruitment process"""
        if not self.results:
            return {'status': 'No workflow executed yet'}
        matching_results = self.results.get('matching_results', {})
        report = {
            'executive_summary': self._generate_executive_summary(),
            'detailed_metrics': self._generate_detailed_metrics(),
            'recommendations': self._generate_recommendations(),
            'matching_results': matching_results,  # <-- ensure matches are always present
            'technical_details': self.results
        }
        return report
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary"""
        if not self.results:
            return {}
        
        workflow_summary = self.results.get('workflow_summary', {})
        
        return {
            'total_candidates_evaluated': workflow_summary.get('total_candidates_processed', 0),
            'qualified_matches_found': workflow_summary.get('matches_found', 0),
            'emails_sent': workflow_summary.get('emails_sent', 0),
            'processing_time': f"{workflow_summary.get('execution_time_seconds', 0):.2f} seconds",
            'success_rate': self._calculate_success_rate()
        }
    
    def _generate_detailed_metrics(self) -> Dict[str, Any]:
        """Generate detailed metrics"""
        if not self.results:
            return {}
        
        matching_results = self.results.get('matching_results', {})
        match_analysis = matching_results.get('match_analysis', {})
        
        return {
            'matching_statistics': match_analysis.get('score_statistics', {}),
            'skill_analysis': match_analysis.get('skill_coverage', {}),
            'agent_performance': self.get_agent_performance()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        if not self.results:
            return ['Execute workflow to get recommendations']
        
        # Check match quality
        matching_results = self.results.get('matching_results', {})
        matches_found = matching_results.get('matches_found', 0)
        
        if matches_found == 0:
            recommendations.append('No matches found - consider lowering similarity threshold')
        elif matches_found < 3:
            recommendations.append('Few matches found - consider expanding search criteria')
        
        # Check processing efficiency
        workflow_summary = self.results.get('workflow_summary', {})
        execution_time = workflow_summary.get('execution_time_seconds', 0)
        
        if execution_time > 60:
            recommendations.append('Consider optimizing processing pipeline for better performance')
        
        return recommendations
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        if not self.results:
            return 0.0
        
        workflow_summary = self.results.get('workflow_summary', {})
        total_candidates = workflow_summary.get('total_candidates_processed', 0)
        matches_found = workflow_summary.get('matches_found', 0)
        
        if total_candidates == 0:
            return 0.0
        
        return (matches_found / total_candidates) * 100
