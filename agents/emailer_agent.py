import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Any, List
from .base_agent import BaseAgent
from utils.helpers import EmailUtils
from config.settings import Config

class EmailerAgent(BaseAgent):
    """Agent responsible for composing and sending emails to matched candidates"""
    
    def __init__(self, config: Config = None):
        super().__init__('emailer', config)
        self.email_utils = EmailUtils()
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for email sending"""
        required_keys = ['matches', 'job_details']
        
        for key in required_keys:
            if key not in input_data:
                self.log_action(f'Missing required key: {key}')
                return False
        
        if not input_data['matches']:
            self.log_action('No matches to send emails to')
            return False
        
        return True
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compose and send emails to top-matching candidates"""
        try:
            self.update_status('processing')
            
            if not self.validate_input(input_data):
                return self.handle_error(ValueError("Invalid input data"), "Input validation failed")
            
            matches = input_data['matches']
            job_details = input_data['job_details']
            email_results = []
            
            for match in matches:
                candidate = match['candidate']
                match_score = match['similarity_score']
                
                # Generate email content
                email_content = self.email_utils.generate_email_template(
                    candidate_name=candidate['name'],
                    position=job_details.get('title', ''),
                    company=job_details.get('company', ''),
                    match_score=match_score,
                    key_skills=candidate.get('extracted_skills', [])
                )
                
                # Send email
                success = self._send_email(
                    recipient_email=candidate['email'],
                    email_content=email_content
                )
                
                email_results.append({
                    'candidate_id': candidate['index'],
                    'candidate_email': candidate['email'],
                    'status': 'sent' if success else 'failed'
                })
                
            self.results = {
                'emails_sent': len(email_results),
                'email_results': email_results
            }
            self.update_status('completed')
            
            return {
                'success': True,
                'results': self.results,
                'agent_info': self.get_info()
            }
            
        except Exception as e:
            return self.handle_error(e, "Email sending failed")
    
    def _send_email(self, recipient_email: str, email_content: str) -> bool:
        """Send email to recipient"""
        try:
            # Validate email
            if not self.email_utils.validate_email(recipient_email):
                self.log_action(f'Invalid recipient email: {recipient_email}')
                return False
            
            # Setup the MIME
            message = MIMEMultipart()
            message['From'] = self.config.EMAIL_ADDRESS
            message['To'] = recipient_email
            message['Subject'] = "Exciting Job Opportunity"
            
            # Add payload to the message
            message.attach(MIMEText(email_content, 'plain'))
            
            # Create SMTP session
            with smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT) as server:
                server.starttls()
                server.login(self.config.EMAIL_ADDRESS, self.config.EMAIL_PASSWORD)
                server.send_message(message)
                self.log_action(f'Email sent to {recipient_email}')
                return True
            
        except Exception as e:
            self.log_action(f'Error sending email to {recipient_email}: {str(e)}')
            return False
            
    def get_email_sending_summary(self) -> Dict[str, Any]:
        """Get summary of email sending results"""
        if not self.results:
            return {'status': 'No emails sent yet'}
        
        return {
            'status': self.status,
            'emails_sent_count': self.results.get('emails_sent', 0),
            'email_results': self.results.get('email_results', [])
        }
