# Evaluate on IGR checkpoint's performance - on test set (let's implement for train set first here)
from .serve import VLLM, get_openai_response
from .graph import self_evaluation
from .dataset.feedback_utils_v2 import Feedback
from typing import Union, List
import torch
from tqdm import tqdm
import pandas as pd

QUERY_THOUGHT_TEMPLATE = """Given query: {query}\nProvide your thought process on how to answer the query."""

class IGREval:
    
    def __init__(self, feedback_content: str, model: str):
        self.feedback = Feedback(content=feedback_content)
        self.use_vllm = torch.cuda.is_available()
        self.MODEL = model
        
        if self.use_vllm:
            from .serve import VLLM
            self.llm = VLLM(name=self.MODEL, gpu_memory_utilization=0.45, temperature=0.8, max_tokens=512, merge=True)
        else:
            print("VLLM not available. Using OpenAI API instead.")
            
    def query_prompt(self, prompt: str):
        completion = "####"
        msg = [
            {"role": "system", "content": self.feedback.content},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ] # This is the standard version of goal-oriented responding template
        chat_str = self.llm.tokenizer.apply_chat_template(msg, tokenize=False)
        return chat_str.split(completion)[0]
    
    def query_thought_prompt(self, prompt: str):
        completion = "####"
        msg = [
            {"role": "system", "content": self.feedback.content},
            {"role": "user", "content": QUERY_THOUGHT_TEMPLATE.format(query=prompt)},
            {"role": "assistant", "content": completion}
        ]
        chat_str = self.llm.tokenizer.apply_chat_template(msg, tokenize=False)
        return chat_str.split(completion)[0]
    
    def query_answer_prompt(self, prompt: str, thought: str):
        completion = "####"
        msg = [
            {"role": "system", "content": self.feedback.content},
            {"role": "user", "content": QUERY_THOUGHT_TEMPLATE.format(query=prompt)},
            {"role": "assistant", "content": thought},
            {"role": "user", "content": "Provide your answer to the query"},
            {"role": "assistant", "content": completion}
        ]
        chat_str = self.llm.tokenizer.apply_chat_template(msg, tokenize=False)
        return chat_str.split(completion)[0]

    def get_response(self, prompts: Union[str, List[str]]):
        if isinstance(prompts, str):  # Support Batch Inference with vLLM
            prompts = [prompts]
        if self.use_vllm:  
            # Generate thought process
            thought_query_prompts = [self.query_thought_prompt(prompt) for prompt in prompts]
            thought_responses = self.llm.completions(thought_query_prompts, max_tokens=500, use_tqdm=True)
            
            # Generate answers based on thought process
            answer_query_prompts = [self.query_answer_prompt(prompt, thought) 
                                    for prompt, thought in zip(prompts, thought_responses)]
            answers = self.llm.completions(answer_query_prompts, max_tokens=500, use_tqdm=True)
            
            # Return only the answers
            return answers
        else:
            raise NotImplementedError("You really wish to evaluate an API based model?")
        
    def evaluate(self, prompts: Union[str, List[str]]):
        responses = self.get_response(prompts)
        
        judges = []
        print("---- Evaluation Begining ----")
        for response in tqdm(responses):
            instruction = self.feedback.content
            try:
                follow_instruction = self_evaluation(instruction, response)
            except: 
                print("Issue in evaluation")
                follow_instruction = False
            judges.append(follow_instruction)
            
        return judges
    
    
    def run(self):
        
        # TODO: Need to be switched to test prompts | Ok this is wrong, we wish to include system prompt into the model 
        # -- We want to enhace its instruction-following ability, given 'no elephant' request, it sticks to the request while providing its response here
        prompts = self.feedback.prompts 
        
        responses = self.get_response(prompts)
        judges = self.evaluate(responses)

        # Create a DataFrame to store the results
        df = pd.DataFrame({
            'prompt': prompts if isinstance(prompts, list) else [prompts],
            'response': responses,
            'is_good': judges
        })

        # Calculate basic statistics
        total_responses = len(judges)
        good_responses = sum(judges)
        bad_responses = total_responses - good_responses
        good_percentage = (good_responses / total_responses) * 100 if total_responses > 0 else 0

        # Print summary statistics
        print(f"Evaluation Complete. Summary Statistics:")
        print(f"Total Responses: {total_responses}")
        print(f"Good Responses: {good_responses}")
        print(f"Bad Responses: {bad_responses}")
        print(f"Good Percentage: {good_percentage:.2f}%")

        # Return both the judges list and the DataFrame
        name_str = self.MODEL.split("/")[-1]
        save_path = f"database/{self.feedback.file_name}/eval_{name_str}.csv"
        df.to_csv(save_path, index=False)
        return df