from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from config.settings import Config, AGENT_ROLES
from utils.helpers import LoggingUtils

class BaseAgent(ABC):
    """Abstract base class for all agents in the system"""
    
    def __init__(self, agent_type: str, config: Config = None):
        self.agent_type = agent_type
        self.config = config or Config()
        self.agent_info = AGENT_ROLES.get(agent_type, {})
        self.name = self.agent_info.get('name', f'{agent_type.title()} Agent')
        self.description = self.agent_info.get('description', '')
        self.capabilities = self.agent_info.get('capabilities', [])
        self.status = 'initialized'
        self.last_action = None
        self.results = {}
        
    def log_action(self, action: str, details: Dict[str, Any] = None) -> None:
        """Log agent actions"""
        details = details or {}
        LoggingUtils.log_agent_action(self.name, action, details)
        self.last_action = {
            'action': action,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
    
    def update_status(self, status: str, details: Dict[str, Any] = None) -> None:
        """Update agent status"""
        self.status = status
        self.log_action(f'Status updated to: {status}', details)
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'type': self.agent_type,
            'name': self.name,
            'description': self.description,
            'capabilities': self.capabilities,
            'status': self.status,
            'last_action': self.last_action
        }
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        pass
    
    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle errors gracefully"""
        error_details = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        self.log_action(f'Error occurred: {error_details["error_type"]}', error_details)
        self.update_status('error', error_details)
        
        return {
            'success': False,
            'error': error_details,
            'results': None
        }
    
    def reset(self) -> None:
        """Reset agent state"""
        self.status = 'initialized'
        self.last_action = None
        self.results = {}
        self.log_action('Agent reset')

class AgentCommunicator:
    """Handles communication between agents"""
    
    def __init__(self):
        self.message_queue = []
        self.active_agents = {}
    
    def register_agent(self, agent_id: str, agent: BaseAgent) -> None:
        """Register an agent for communication"""
        self.active_agents[agent_id] = agent
    
    def send_message(self, from_agent: str, to_agent: str, message: Dict[str, Any]) -> None:
        """Send message between agents"""
        message_obj = {
            'from': from_agent,
            'to': to_agent,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.message_queue.append(message_obj)
    
    def get_messages(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get messages for a specific agent"""
        return [msg for msg in self.message_queue if msg['to'] == agent_id]
    
    def broadcast_message(self, from_agent: str, message: Dict[str, Any]) -> None:
        """Broadcast message to all agents"""
        for agent_id in self.active_agents:
            if agent_id != from_agent:
                self.send_message(from_agent, agent_id, message)

class AgentRegistry:
    """Registry for managing all agents"""
    
    def __init__(self):
        self.agents = {}
        self.communicator = AgentCommunicator()
    
    def register_agent(self, agent_id: str, agent: BaseAgent) -> None:
        """Register a new agent"""
        self.agents[agent_id] = agent
        self.communicator.register_agent(agent_id, agent)
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """Get all registered agents"""
        return self.agents.copy()
    
    def get_agents_by_capability(self, capability: str) -> List[BaseAgent]:
        """Get agents that have a specific capability"""
        return [agent for agent in self.agents.values() 
                if capability in agent.capabilities]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'total_agents': len(self.agents),
            'agents_status': {
                agent_id: agent.get_info() 
                for agent_id, agent in self.agents.items()
            },
            'message_queue_size': len(self.communicator.message_queue)
        }
