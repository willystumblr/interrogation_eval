from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any, Dict, List
from src.schemas import Action
import logging

class Agent(ABC):
    def __init__(self, role: str, **kwargs):
        self.role = role
        self.system_message = kwargs.get('system_message', "")
        self.memory: List = [
            {"role": "system", "content": self.system_message}
        ]
        if 'model' in kwargs:
            self.model = kwargs['model']
        else:
            logging.warning("No model specified for the agent. Setting model to the default value: 'gemini/gemini-2.5-flash'.")
            self.model = 'gemini/gemini-2.5-flash'

    @abstractmethod
    def act(self, message: str | Dict | Any) -> Action:
        """Decide on an action based on the current state."""
        pass
    
    def reset(self):
        """Reset the agent's internal state."""
        self.memory = self.memory[:1]  # keep only the system message
    
    