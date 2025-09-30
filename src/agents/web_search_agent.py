from typing import Dict
from src.agents.base_agent import Agent
from src.schemas import Action, ToolCall
from src.utils import get_completion
import logging
import json
import time

class WebSearchAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role=kwargs.get('role', "web_search"),
            system_message=kwargs.get('system_message', "")
        )
        self.tools = kwargs.get('tools', [])
        self.cutoff_date = time.strftime("%Y-%m-%d") # default to current date
    
    def set_cutoff_date(self, cutoff_date: str) -> None:
        self.cutoff_date = cutoff_date
    
    def act(self, message: Dict) -> Action:
        if not self.tools:
            return Action(
                action_type="next_agent",
                target_agent="questioner"
            )
        # for entity_claim in message:
        entity = message.get('entity', 'unknown entity')
        claim = message.get('claim', 'unknown claim')
        rationale = message.get('rationale', '')
        if entity != 'unknown entity' and claim != 'unknown claim':
            while True:
                res = get_completion(
                    model=self.model,
                    messages=self.memory + [{"role": "user", "content": f"Does the following claim-entity pair need to be verified with web-search?\nClaim: {claim}\nEntity: {entity}\nCutoff Date: {self.cutoff_date}"}],
                    tool_choice="none",
                    tools=self.tools,
                    reasoning_effort="low",
                )
                res_ = res.choices[0].message.content.lower()
                if res_ in ['yes', 'no']:
                    break

        if res_ == 'yes':
            res = get_completion(
                model=self.model,
                messages=self.memory + [{"role": "user", "content": f"Given the entity: {entity}, and the claim: {claim}, with rationale: {rationale}, decide the best tool to use to verify the claim."}],
                tool_choice="required",
                tools=self.tools,
                reasoning_effort="low",
            )
            res_ = res.choices[0].message.model_dump()
            if 'tool_calls' in res_ and res_['tool_calls']:
                tool_call = res_['tool_calls'][0]
                tool_name = tool_call['function']['name']
                arguments = json.loads(tool_call['function']['arguments'])
                return Action(
                    agent=self.role,
                    action_type="tool_call",
                    tool_call=ToolCall(
                        tool_name=tool_name,
                        arguments=arguments,
                        details=res_
                    )
                )
        else:
            logging.warning(f"WebSearchAgent decided no web search needed for entity: {entity}, claim: {claim}.")
            return None