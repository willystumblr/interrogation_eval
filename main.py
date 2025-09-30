import os
from src.utils import setup_logging, read_json, write_json
from src.env.interrogation_env import InterrogationEnv
from src.tools.web_search import GoogleClaimSearch
from src.tools.address_locator import GoogleGeocodeValidate
from dotenv import load_dotenv
from datasets import load_dataset
import argparse
import re
import time
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Run the interrogation environment.")
    parser.add_argument('--baseline_name', type=str, required=True, help='Baseline name for the interviewee simulator.')
    parser.add_argument('--user_id', type=str, default=None, help='User ID for Character AI.')
    parser.add_argument('--num_turns', type=int, default=30, help='Maximum number of turns in the interrogation.')
    parser.add_argument('--sample', action='store_true', help='Whether to sample OpenCharacter personas.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling personas.')
    parser.add_argument('--log_to_file', action='store_true', help='Whether to log to a file.')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_to_file=args.log_to_file, process_name="main")
    load_dotenv()
    
    interviewee_kwargs = []
    # set up baseline interviewee simulator
    if args.baseline_name == "characterai":
        assert args.user_id is not None, "Character AI requires user_id parameter"
        personas = read_json("src/env/personas/characterai.json")
        for persona in personas:
            interviewee_kwargs.append({
                "baseline_name": "characterai",
                "character_id": persona['character_id'],
                "user_id": args.user_id,
                "name": persona['character_name']
            })    
    elif args.baseline_name == "human_simulacra":
        interviewee_kwargs = [{
            "baseline_name": "human_simulacra",
            "name": name
        } for name in ["Mary Jones", "Haley Collins", "Sara Ochoa", "James Jones", "Tami Clark", "Michael Miller", "Kevin Kelly", "Erica Walker", "Leslie Nichols", "Robert Scott", "Marsh Zhaleh"]]
    elif args.baseline_name == "opencharacter":
        dataset = load_dataset("xywang1/OpenCharacter", "Synthetic-Character", split="train")
        if args.sample:
            dataset = dataset.shuffle(seed=args.seed).select(range(10))
        for data in dataset:
            name_match = re.match(r"Name:\s(.*)\n",  data['character'])
            if not name_match:
                logging.warning(f"Could not extract name from character profile: {data['character']}. Skipping this persona.")
                continue
            interviewee_kwargs.append({
                "baseline_name": "opencharacter",
                "model_path": "willystumblr/opencharacter-sft-2025-06-21_14-54-13", # hardcoded for now
                "persona": data['persona'],
                "profile": data['character'],
                "name": name_match.group(1).strip(),
                "load_in_4bit": True
            })
    elif args.baseline_name == "human_interview":
        interviewee_kwargs = [{
            "baseline_name": "human_interview",
            "name": input("Enter your name: ")
        }]
    else:
        raise ValueError("Invalid baseline name. Choose from ['characterai', 'human_simulacra', 'opencharacter', 'human_interview']")
    
    for interviewee_kwarg in interviewee_kwargs:
        try:
            logging.info(f"Starting new session with interviewee: {interviewee_kwarg.get('name', 'unknown')}, baseline: {interviewee_kwarg['baseline_name']}")
            env = InterrogationEnv(
                tools={
                    "google_claim_search": GoogleClaimSearch(
                        api_key=os.getenv('GOOGLE_API_KEY'),
                        cx=os.getenv('GOOGLE_CX')
                    ),
                    "google_geocode_validate": GoogleGeocodeValidate(api_key=os.getenv('GOOGLE_API_KEY'))
                },
                max_turns=args.num_turns,
                **interviewee_kwarg
            )
            state = env.reset()
            done = False
            while not done:
                state, done = env.step()
            state = env.finalize()
            result_path = f"data/results/{args.baseline_name}/{interviewee_kwarg.get('name', 'unknown').replace(' ', '_')}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
            env.save_state(result_path)
        except Exception as e:
            logging.exception(f"Error during session with interviewee {interviewee_kwarg.get('name', 'unknown')}, baseline: {interviewee_kwarg['baseline_name']}: {e}")
            continue
