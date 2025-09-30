from typing import List, Dict
import re
import time
from pydantic import BaseModel, Field, ValidationError
import logging
import litellm
import os
import json
import time
from src.agents.base_agent import Agent
from src.utils import get_completion
from src.schemas import Action, Observation
litellm.drop_params = True

class QuestionerAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role=kwargs.get('role', "questioner"),
            system_message=kwargs.get('system_message', ""),
            model=kwargs.get('model', "gemini/gemini-2.5-flash")
        )

    def set_cutoff_date(self, cutoff_date: str) -> None:
        self.memory[0]['content'] = self.memory[0]['content'].format(current_date=cutoff_date)

    def update_memory(self, **kwargs) -> None:
        self.memory.append(kwargs) # typically role and content

    def act(self, observation: Observation, **kwargs) -> Action:
        if observation.observation_type == "tool_output":
            assert 'tool_calls' in self.memory[-1] and self.memory[-1]['tool_calls'] is not None, "Last memory entry must be a tool call."
            self.update_memory(
                role="tool",
                tool_call_id=self.memory[-1]['tool_calls'][0]['id'],
                name=self.memory[-1]['tool_calls'][0]['function']['name'],
                content=str([output.model_dump() for output in observation.tool_output])
            )
        res = get_completion(
            model=self.model,
            messages=self.memory,
            reasoning_effort="low"
        )
        question = res.choices[0].message.content.strip()
        self.update_memory(role="assistant", content=question)
        return Action(action_type="respond", content=question)

