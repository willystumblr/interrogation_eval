from PyCharacterAI import get_client
from PyCharacterAI.exceptions import SessionClosedError
import asyncio
import gc
import torch
import logging
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.env.personas.human_simulacra.hs_agents import Top_agent
from src.utils import get_completion
from src.schemas import Action, IntervieweeResponse
import time
import nest_asyncio
from dotenv import load_dotenv
import os

nest_asyncio.apply()

class IntervieweeSimulator:
    def __init__(self, **kwargs):
        load_dotenv()
        assert kwargs.get('baseline_name') in ["characterai", "human_simulacra", "opencharacter", "human_interview"], "Invalid baseline name"
        self.type = kwargs.get('baseline_name')
        self.name = kwargs.get('name', None)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.__nhd_prompt = open(f"{project_root}/src/agents/prompts/nhd_detector.txt", "r").read()
        
        if self.type == "characterai":
            #### **character_id, user_id, name** are required ####
            
            assert 'character_id' in kwargs, "Character AI requires character_id parameter"
            assert 'user_id' in kwargs, "Character AI requires user_id parameter"
            
            self.char_id = kwargs['character_id']
            self.user_id = kwargs['user_id']
            
            try:
                asyncio.run(self._setup_client_and_chat(kwargs['user_id'], kwargs['character_id']))
                assert self.chat_id is not None, "Chat ID must be set after setup"
                logging.info(f"CharacterAI client and chat session established. Chat ID: {self.chat_id}")
            except SessionClosedError as e:
                logging.error(f"Session closed error: {e}")
                self.client_or_model = None
                self.chat_id = None
            
        
        elif self.type == "human_simulacra":
            #### **name** are required
            assert 'name' in kwargs, "Human Simulacra requires name parameter"
            self.client_or_model = Top_agent(character_name=kwargs['name']) ### has its own chat history
            
        elif self.type == "opencharacter": 
            #### **model_path, persona, profile** are required ####
            
            assert 'model_path' in kwargs, "OpenCharacter requires (path-like, either huggingface repo OR local path) model parameter"
            assert 'persona' in kwargs, "OpenCharacter requires persona parameter"
            assert 'profile' in kwargs, "OpenCharacter requires profile parameter"
            
            self.client_or_model = AutoModelForCausalLM.from_pretrained(
                kwargs['model_path'],
                load_in_4bit=kwargs.get('load_in_4bit', False), # default is False
                # device_map={"":0}
            ).to("cuda").eval()
            self.tokenizer = AutoTokenizer.from_pretrained(kwargs['model_path'])
            self.history = [{
                "role": "system",
                "content": ("You are an AI character with the following Persona.\n\n"
                    f"# Persona\n{kwargs['persona']}\n\n"
                    f'# Character Profile\n{kwargs["profile"]}\n\n'
                    "Please stay in character, be helpful and harmless."
                ),
            }]
            self.name = re.search(r'^Name:\s*(.+)$', kwargs['profile'], flags=re.MULTILINE).group(1).strip() if self.name is None else self.name
        else: # human_interview
            pass
            
    def get_response(self, message: str) -> IntervieweeResponse:
        if self.type == "characterai":
            # Add a small delay before sending message
            time.sleep(0.5)  # 500ms delay
           
            response = asyncio.run(
                self.client_or_model.chat.send_message(
                    character_id=self.char_id, 
                    chat_id=self.chat_id, 
                    text=message
                )
            )
            response = response.get_primary_candidate().text
        
        elif self.type == "human_simulacra":
            response = self.client_or_model.send_message(message)
        
        elif self.type == "opencharacter": # OpenCharacter
            self.history.append({
                "role": "user",
                "content": message
            })
            
            while True:
                input_ids = self.tokenizer.apply_chat_template(
                    self.history,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                ).to(self.client_or_model.device)

                if input_ids.shape[1] <= self.client_or_model.config.max_position_embeddings:
                    break

                # drop oldest assistant-user pair but keep system prompt
                if len(self.history) > 3:
                    self.history = [self.history[0]] + self.history[3:]
                else:
                    # still too long even after pruning – fallback
                    self.history = [self.history[0]] + self.history[-2:]

            with torch.no_grad():
                output_ids = self.client_or_model.generate(
                    input_ids,
                    max_new_tokens=1024, # following the original config
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.9,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            response = self.tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
            self.history.append({
                "role": "assistant",
                "content": response
            })
            
        else:  # human_interview
            response = input(f"Your Response: ")
        
        # nhd
        while True:
            res = get_completion(
                model="gemini/gemini-2.5-flash",
                messages=[{"role":"system", "content": self.__nhd_prompt}, {"role":"user", "content": f"Interviewer:{message}\nInterviewee: {response}"}],
                reasoning_effort="low",
                temperature=0.0
            )
            res_ = res.choices[0].message.content.strip()
            if res_ in ['### PASS ###', '### FAIL ###']:
                break
        if res_ == '### FAIL ###':
            logging.warning("AI Detected! Terminating the interview: " + response)
            raise ValueError("AI Detected")
        
        return IntervieweeResponse(
            question=message,
            content=response
        )
   
    async def _setup_client_and_chat(self, user_id: str, char_id: str ):
        try:
            self.client_or_model = await get_client(user_id)
            
            me_task = asyncio.create_task(self.client_or_model.account.fetch_me())
            chat_task = asyncio.create_task(self.client_or_model.chat.create_chat(char_id))
            
            me, (chat, greeting_message) = await asyncio.gather(me_task, chat_task)
            
            self.chat_id = chat.chat_id
            
        except Exception as e:
            logging.error(f"Failed to set up the cai client: {e}")
            await self.client_or_model.close_session()
    def clear_model(self):
        if self.type == "opencharacter": # no action needed for other types
            del self.client_or_model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    async def close(self):
        if self.type == "characterai":
            if hasattr(self, 'client_or_model') and self.client_or_model is not None:
                await self.client_or_model.close_session()
                self.client_or_model = None
                self.chat_id = None
                