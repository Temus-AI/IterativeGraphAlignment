from .dataset.feedback_utils_v2 import Feedback
from .serve import get_openai_response
from tqdm import tqdm
from typing import Union, List
import time
import random
import torch
from .config import *
from src.IGRlearn import IGReasoner, StarReasoner, Rationale, convert_graph_to_string, format_qa_prompt, prepare_propose_prompt, prepare_parrot_prompt, prepare_naive_prompt
from src.graph import iterative_graph_prompting
import glob
from datasets import load_dataset

# Rationale: Model collapse when learning from its own synthetic dataset 
# -- we want mutation from another model
# -- maynot need it from a very strong model necessarily

def apply_roleplay_transform(query_str):
    
    # pattern specification
    replace_pattern = [("I should ASK", "I should ask"), "I should REJECT"]
    if_then_add_pattern = ["Customer can ask", "I should REJECT current question"]
    
    # Replace patterns
    for pattern in replace_pattern[0]:
        query_str = query_str.replace(pattern, replace_pattern[1])
    
    # If-then-add patterns
    if if_then_add_pattern[0] in query_str: 
        query_str += f"\n{if_then_add_pattern[1]}"
    return query_str 

# Missing Piece: for each iteration, our SLM should be different | BLM should also varies (We want to consult different people for our problem ....)
# -- Need iter_number, as well as the previous iteration model name (on HF potentially) so we don't need to store it locally
from src.config import IGRConfig

class IGRSampler:

    def __init__(self, 
                 feedback_content: str, 
                 config: IGRConfig,
                 weak_anno=True,
                 load_slm=True,
                 helper_idx = 0,
                 roleplay=True,
                 init_naive=True,
                 do_star=False,
                 SLM_MODEL: str = ""):

        self.use_vllm = torch.cuda.is_available()
        if SLM_MODEL:
            self.SLM_MODEL = SLM_MODEL
        else:
            self.SLM_MODEL = config.BASE_MODEL
        self.BLM_MODEL = config.HELPER_MODEL[int(helper_idx % len(config.HELPER_MODEL))]  
        self.id = config.ID
        self.current_iteration = config.current_iteration
        self.init_naive = init_naive
        self.load_slm = load_slm
        self.roleplay = roleplay
        self.do_star = do_star
        
        self.feedback = Feedback(content=feedback_content, weak_anno=weak_anno)
        self.rationale_dir = f'database/{self.feedback.file_name}/Sample_Rationale_{self.id}_iter{self.current_iteration}'
        self.star_rationale_dir = f'database/{self.feedback.file_name}/STaR_Rationale_{self.id}_iter{self.current_iteration}'
    
        self.eval_dir = ""
        os.makedirs(self.rationale_dir, exist_ok=True)
        
        if self.use_vllm:
            from .serve import VLLM
            if self.load_slm:
                self.slm = VLLM(name=self.SLM_MODEL, gpu_memory_utilization=0.85, temperature=0.8, max_tokens=512, merge=True) # SLM includes merged checkpoints
                self.blm = False 
            else:
                self.blm = VLLM(name=self.BLM_MODEL, gpu_memory_utilization=0.9, temperature=0.8, max_tokens=512, merge=False) # BLM is always helper models
                self.slm = False 
            self.use_vllm = True
        else:
            print("VLLM not available. Using OpenAI API instead.")
            self.use_vllm = False
            if self.load_slm:
                self.slm = "gpt-4o-mini"
                self.blm = False
            else:
                self.blm = "gpt-4o"
                self.slm = False

    def get_slm_response(self, prompts: Union[str, List[str]]):
        if not self.slm:
            raise AssertionError("SLM (Student Language Model) is not initialized.")
        
        if isinstance(prompts, str): # Support Batch Inference with vLLM
            prompts = [prompts]
            
        if self.use_vllm:
            return self.slm.completions(prompts, max_tokens=500, use_tqdm=True)
        else:
            responses = []
            for prompt in tqdm(prompts):
                responses.append(get_openai_response(prompt, "gpt-4o-mini"))
            return responses

    def get_blm_response(self, prompts: Union[str, List[str]]):
        if not self.blm:
            raise AssertionError("BLM (Helper Language Model) is not initialized.")
        
        if isinstance(prompts, str): # Support Batch Inference with vLLM
            prompts = [prompts]
            
        if self.use_vllm:
            return self.blm.completions(prompts, max_tokens=500, use_tqdm=True)
        else:
            responses = []
            for prompt in tqdm(prompts):
                responses.append(get_openai_response(prompt, "gpt-4o"))
            return responses
    
    @property
    def prompt_tuples(self):
        if self.feedback.content in ["Provide your thought and answer to the question from user. For example:\n        Thought: ...\n        Answer: ..."]:  # This is fine
            # Implement logic to obtain propose_prompt, correct_answer, parrot_prompt, naive_prompt respectively
            gsm_ds = load_dataset("Ksgk-fy/gsm8k")
            prompt_tuples = []
            for d in gsm_ds["train"]:
                correct_answer = d["assistant"]
                prompt = d["user"]
                instruction = d["system"]
                hint = d['thought']
                naive_prompt = prepare_naive_prompt(instruction, prompt)
                propose_prompt = prepare_propose_prompt(instruction, prompt, hint)
                parrot_prompt = prepare_parrot_prompt(instruction, prompt, hint, correct_answer)
                prompt_tuples.append((prompt, propose_prompt, correct_answer, parrot_prompt, naive_prompt))
            return prompt_tuples

        prompt_tuples = []
        for prompt in self.feedback.prompts:
            try:
                _, graph = self.feedback.load_graph_and_image(prompt)
            except:
                print("Prompt missing graph, generating now ...  || ", prompt)
                iterative_graph_prompting(self.feedback, prompt, model="claude", roleplay=True)
                _, graph = self.feedback.load_graph_and_image(prompt)
            
            graph_str = convert_graph_to_string(graph) # need to be fixed for roleplay scenario 
            if self.roleplay:
                graph_str = apply_roleplay_transform(graph_str)
                       
            prompt_str = prompt.replace(" ", "-")
            correct_answer = self.feedback.correct_responses[prompt_str]
            igp_answer = self.feedback.load_response(prompt, mode="IGP", model="claude")
            if self.do_star:
                correct_answer = igp_answer 
            naive_prompt = prepare_naive_prompt(self.feedback.content, prompt)
            propose_prompt = prepare_propose_prompt(self.feedback.content, prompt, graph_str)
            parrot_prompt = prepare_parrot_prompt(self.feedback.content, prompt, graph_str, igp_answer)
            prompt_tuples.append((prompt, propose_prompt, correct_answer, parrot_prompt, naive_prompt)) # Parrot prompt used to deal with hard cases
        return prompt_tuples
    
    @property
    def prompt_tuples_eval(self):
        if self.feedback.content in ["Provide your thought and answer to the question from user. For example:\n        Thought: ...\n        Answer: ..."]:  # This is fine
            # Implement logic to obtain propose_prompt, correct_answer, parrot_prompt, naive_prompt respectively
            gsm_ds = load_dataset("Ksgk-fy/gsm8k")
            prompt_tuples = []
            for d in gsm_ds["test"]:
                correct_answer = d["assistant"]
                prompt = d["user"]
                instruction = d["system"]
                hint = d['thought']
                naive_prompt = prepare_naive_prompt(instruction, prompt)
                propose_prompt = prepare_propose_prompt(instruction, prompt, hint)
                parrot_prompt = prepare_parrot_prompt(instruction, prompt, hint, correct_answer)
                prompt_tuples.append((prompt, propose_prompt, correct_answer, parrot_prompt, naive_prompt))
            return prompt_tuples
        
        return self.prompt_tuples[self.feedback.num_train:]
    
    def postprocess_igr_data(self, igr_dataset, prompt):
        """ 
        Replace hint-included user query with hint-free system prompt + user prompt
        """
        processed_dataset = []
        for message in igr_dataset:
            # Remove first element 
            message = message[1:]
            # Get Instruction & Query and slot them into the start of the message
            instruction, query = self.feedback.content, prompt
            system_prompt = {"role": "system", "content": instruction}
            query_prompt = {"role": "user", "content": query}
            message = [system_prompt, query_prompt] + message
            processed_dataset.append(message)
            
        return processed_dataset
    
    def eval_prompts(self, iter_n: int = 1, check_alignment=True):
        
        if self.use_vllm:
            apply_chat_template_slm = lambda msg: self.slm.tokenizer.apply_chat_template(msg, tokenize=False)
        else:
            apply_chat_template_slm = lambda msg: msg 
            
        self.slm_thinker = IGReasoner.make_from_tuples(self.get_slm_response, self.prompt_tuples_eval, apply_chat_template=apply_chat_template_slm, n_answer_per_question=1)
        
        df_eval = self.slm_thinker.eval(check_alignment=check_alignment)
        
        if not self.eval_dir:
            self.eval_dir = f"database/{self.feedback.file_name}/Evaluation_{self.id}_iter{self.current_iteration}.csv"
            
        df_eval.to_csv(self.eval_dir, index=False)
        
    def process_prompts(self, iter_n = 2, n_think=5, n_parrot=10, check_alignment=True): # This function needs to be modified, specifically prepare_star_datapoints needs to be changed here....
            
        # Naive Responding with Base Model
        if self.slm and self.init_naive:
            
            if self.use_vllm:
                apply_chat_template_slm = lambda msg: self.slm.tokenizer.apply_chat_template(msg, tokenize=False)
            else:
                apply_chat_template_slm = lambda msg: msg 
                
            self.slm_thinker = IGReasoner.make_from_tuples(self.get_slm_response, self.prompt_tuples, apply_chat_template=apply_chat_template_slm, n_answer_per_question=n_think)
            
            print("Base model naive responding ...")
            self.slm_thinker.think(include_failed=True, do_parrot=False, do_naive=True, check_alignment=check_alignment)
            
            self.slm_thinker.save_rationales(self.rationale_dir) # Assume pre-specified rationale dir (one level above)
            
        # Hint-Included Responding with Base Model
        if self.slm and not self.init_naive:
            
            if self.use_vllm:
                apply_chat_template_slm = lambda msg: self.slm.tokenizer.apply_chat_template(msg, tokenize=False)
            else:
                apply_chat_template_slm = lambda msg: msg 
                
            self.slm_thinker = IGReasoner.load_from_rationales(self.get_slm_response, self.rationale_dir, apply_chat_template=apply_chat_template_slm, multiplication_factor=n_think)

            print("Base model thinking ...")
            self.slm_thinker.think(include_failed=True, do_parrot=False, do_naive=False, check_alignment=check_alignment)
            
            self.slm_thinker.save_rationales(self.rationale_dir)

        # Hint-Included / Parrot Responding with Helper Model
        if self.blm: 

            if self.use_vllm:
                apply_chat_template_blm = lambda msg: self.blm.tokenizer.apply_chat_template(msg, tokenize=False)
            else:
                apply_chat_template_blm = lambda msg: msg 
                
            self.blm_thinker = IGReasoner.load_from_rationales(self.get_blm_response, self.rationale_dir, apply_chat_template = apply_chat_template_blm, multiplication_factor = n_parrot)
            
            print("Helper model thinking ...")
            self.blm_thinker.think(include_failed=True, do_parrot=False, check_alignment=check_alignment)
            
            print("\nHelper model parroting ...")
            self.blm_thinker.think(include_failed=True, do_parrot=True, check_alignment=check_alignment)
            
            self.blm_thinker.save_rationales(self.rationale_dir) # save again, just for fun

            print("Saving rationales collected ...")
            for prompt in self.feedback.prompts: 
                igr_dataset = self.blm_thinker.igr_data(prompt)
                if len(igr_dataset) == 0:
                    print("No rationale obtained for current prompt: ", prompt)
                    continue 
                processed_dataset = self.postprocess_igr_data(igr_dataset, prompt) # Post Processing
                if len(processed_dataset) > 0:
                    self.feedback.save_IGR_dataset(prompt, processed_dataset, iter_n=iter_n, ID=self.id)
                    
    def process_mistake_prompts(self, require_wrong_answers=3):
        
        print("Inspection on condition: slm: {self.slm} | init_naive: {self.init_naive}")
        
        if self.use_vllm:
            apply_chat_template_slm = lambda msg: self.slm.tokenizer.apply_chat_template(msg, tokenize=False)
        else:
            apply_chat_template_slm = lambda msg: msg 
        
        self.slm_thinker = IGReasoner.make_from_tuples(self.get_slm_response, self.prompt_tuples, apply_chat_template=apply_chat_template_slm, n_answer_per_question=require_wrong_answers)
        
        self.slm_thinker.make_mistakes(batch_size=5000)
        
        self.slm_thinker.make_mistakes(batch_size=5000)
        
        self.slm_thinker.save_rationales(self.star_rationale_dir)  
    
# Function to get the correct letter based on the correct answer
def get_correct_letter(options, correct_answer):
    for i, option in enumerate(options):
        if option == correct_answer:
            return chr(65 + i)  # 'A', 'B', 'C', or 'D'
    return None
    
def prepare_star_prompt(prompt, rationales):

    wrong_answers = []
    for rationale in rationales:
        correct_answer = rationale.correct_answer 
        
        if rationale.prompt == prompt and rationale.is_wrong:
            wrong_answers.append(rationale.answer)
        if len(wrong_answers) == 0:
            wrong_answers = ["Are you Bruce Wayne?"] * 3
        if 0 < len(wrong_answers) < 3:
            wrong_answers = (wrong_answers * 3)[:3]
    # 
    SINGLE_CHOICE_PROMPT_TEMPLATE = """Given the following options, select the correct answer to the following question.
    Question: {question}
    A: {A}
    B: {B}
    C: {C}
    D: {D}
    Provide your thought and choice, for example:
    Thought: [Your thinking process]
    Answer: [Your answer]
    """
    
    EXPLAIN_CHOICE_PROMPT_TEMPLATE = """Given the following options, explain why the answer {answer} is correct to the following question.
    Question: {question}
    A: {A}
    B: {B}
    C: {C}
    D: {D}
    Provide your thought, for example:
    Thought: [Your thinking process]
    """
    
    
    # Prepare the single choice prompt
    options = [correct_answer] + wrong_answers[:3]
    import random
    random.shuffle(options)

    single_choice_prompt = SINGLE_CHOICE_PROMPT_TEMPLATE.format(
        question=prompt,
        A=options[0],
        B=options[1],
        C=options[2],
        D=options[3]
    )

    correct_letter = get_correct_letter(options, correct_answer)
    
    explain_choice_prompt = EXPLAIN_CHOICE_PROMPT_TEMPLATE.format(
        question=prompt,
        A=options[0],
        B=options[1],
        C=options[2],
        D=options[3],
        answer=correct_letter
    )
    
    return single_choice_prompt, correct_letter, explain_choice_prompt


class StarSampler:
    
    def __init__(self, 
                 feedback_content: str, 
                 config: IGRConfig,
                 weak_anno=True):
        
        self.feedback = Feedback(content=feedback_content, weak_anno=weak_anno)
        self.use_vllm = torch.cuda.is_available()
        self.SLM_MODEL = config.BASE_MODEL
        self.id = config.ID
        self.current_iteration = config.current_iteration
        self.star_rationale_dir = f'database/{self.feedback.file_name}/STaR_Rationale_{self.id}_iter{self.current_iteration}'
        self.star_choice_dir = f'database/{self.feedback.file_name}/STaR_Choice_{self.id}_iter{self.current_iteration}'
        os.makedirs(self.star_rationale_dir, exist_ok=True)
        
        if self.use_vllm:
            from .serve import VLLM
            self.slm = VLLM(name=self.SLM_MODEL, gpu_memory_utilization=0.85, temperature=0.8, max_tokens=512, merge=True) # SLM includes merged checkpoints
        else:
            self.slm = "gpt-4o-mini"
            

    def get_slm_response(self, prompts: Union[str, List[str]]):
        if not self.slm:
            raise AssertionError("SLM (Student Language Model) is not initialized.")
        
        if isinstance(prompts, str): # Support Batch Inference with vLLM
            prompts = [prompts]
            
        if self.use_vllm:
            return self.slm.completions(prompts, max_tokens=500, use_tqdm=True)
        else:
            responses = []
            for prompt in tqdm(prompts):
                responses.append(get_openai_response(prompt, "gpt-4o-mini"))
            return responses
    
    @property
    def prompt_tuples(self):
        files = glob.glob(self.star_rationale_dir + "/*")
        rationales = []
        for file_path in files:
            rationales.append(Rationale.load(file_path))
            
        prompt_tuples = []
        for prompt in self.feedback.prompts:                       
            star_prompt, correct_letter, explain_choice_prompt = prepare_star_prompt(prompt, rationales)
            prompt_tuples.append((prompt, star_prompt, correct_letter, explain_choice_prompt))
        return prompt_tuples
    
        
    def sample(self, iter_n=1):
        
        if self.use_vllm:
            apply_chat_template_slm = lambda msg: self.slm.tokenizer.apply_chat_template(msg, tokenize=False)
        else:
            apply_chat_template_slm = lambda msg: msg 
            
        self.star_thinker = StarReasoner.make_from_tuples(self.feedback.content, self.get_slm_response, self.prompt_tuples, apply_chat_template=apply_chat_template_slm, n_answer_per_question=10)
        
        print("Base model STaR reasoning ...")
        self.star_thinker.think(do_choice=True)
        
        print("Backward Rationalization ...")
        self.star_thinker.think(do_choice=False)
        
        self.star_thinker.save_choices(self.star_choice_dir)

        print("Saving STaR rationales collected ...")
        for prompt in self.feedback.prompts[self.feedback.num_train:]: 
            star_dataset = self.star_thinker.star_data(prompt)
            if len(star_dataset) > 0:
                self.feedback.save_STaR_dataset(prompt, star_dataset, iter_n=iter_n, ID=self.id)    