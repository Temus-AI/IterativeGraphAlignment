import glob 
import os
import json
import networkx as nx
from typing import List, Tuple, Optional
import openai 
from openai import OpenAI 
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from .serve import get_helper_response
import time 
from dataclasses import dataclass 
from .config import *

# Temporary function to get response from OpenAI | We need a SLM vLLM endpoint here instead
def get_oai_response(prompt: str):
    client = OpenAI()
    model_name = "gpt-4o"

    completion = client.chat.completions.create(
        model = model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        stop=None
    )
    response = completion.choices[0].message.content
    return response 

# Dataset Curation

def traverse_graph(graph: nx.DiGraph, start_node=None) -> List[List[str]]:
    """ 
    Given a Graph and a start_node (default to the first node in the graph)
    Return a list of paths in the graph which begin with the first node, including multi-hop connections
    """
    if start_node is None:
        start_node = list(graph.nodes())[0]
    
    paths = []
    
    def dfs(node, current_path):
        paths.append(current_path)
        for neighbor in graph.neighbors(node):
            if neighbor not in current_path:
                dfs(neighbor, current_path + [neighbor])
            elif neighbor == start_node and len(current_path) > 2:
                # Add cycle back to start node if it exists
                paths.append(current_path + [neighbor])
    
    dfs(start_node, [start_node])
    return paths 

def prepare_IGR_data(graph: nx.DiGraph, paths: List[List[str]]) -> List[Tuple[List[str], str, List[str]]]:
    """ 
    Given a graph and a list of Paths, which contains a chain of thinking processes
    Return a list of tuples, which contains (pre-condition, reasoning, hint)
    For instance for path A ->relation-> B, the tuples would be ([], A relation B, [])
    For path A -> relation1 -> B -> relation2 -> C, the tuples would be:
    - ([A relation1 B], B relation2 C, [])
    - If A -> C exists: ([], A relation3 C, [A relation1 B, B relation2 C])
    """
    triplets = []
    for path in paths:
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i+1]
            relation = graph[current_node][next_node]['label']
            reasoning = f"{current_node} {relation} {next_node}"
            
            # Direct connection
            pre_condition = []
            if i > 0:
                pre_condition = [f"{path[j]} {graph[path[j]][path[j+1]]['label']} {path[j+1]}" for j in range(i)]
            
            hint = []  # Initialize hint as an empty list
            triplets.append((pre_condition, reasoning, hint))
                
            # Check for multi-hop connections
            for j in range(i+2, len(path)):
                multi_hop_node = path[j]
    
                if graph.has_edge(current_node, multi_hop_node):
                    multi_hop_relation = graph[current_node][multi_hop_node]['label']
                    multi_hop_reasoning = f"{current_node} {multi_hop_relation} {multi_hop_node}"
                    multi_hop_hint = [
                        f"{path[k]} {graph[path[k]][path[k+1]]['label']} {path[k+1]}" 
                        for k in range(i, j)
                    ]
                    multi_hop_pre_condition = []  # Multi-hop connections start with empty pre-condition
                    triplets.append((multi_hop_pre_condition, multi_hop_reasoning, multi_hop_hint))
    
    return triplets


def remove_duplicate_triplets(triplets: List[Tuple[List[str], str, List[str]]]) -> List[Tuple[List[str], str, List[str]]]:
    # Remove duplicate triples
    unique_triplets = []
    seen = set()

    for triplet in triplets:
        # Convert the triplet to a hashable format
        hashable = (
            tuple(triplet[0]),  # Convert list to tuple
            triplet[1],
            tuple(tuple(x) for x in triplet[2])  # Convert list of lists to tuple of tuples
        )
        
        if hashable not in seen:
            seen.add(hashable)
            unique_triplets.append(triplet)
    
    return unique_triplets

# Need to change the code starting from this line here

QAH_TEMPLATE = """Given the instruction: {instruction}

Query: {prompt}

Hint: {hint}

Proposed answer: {answer}

Task: Evaluate if the proposed answer follows the given instruction.

Please provide your evaluation in the following format:
Rationale: [Explain your reasoning here, considering how well the answer aligns with the instruction]
Answer: [True if the answer follows the instruction, False if it doesn't]"""


# MVP approach: Print out the entire graph as strings "Entity1 Relation12 Entity2"
def convert_graph_to_string(graph):
    edge_strings = [f"{u} {d['label']} {v}" for u, v, d in graph.edges(data=True)]
    return "\n".join(edge_strings)

def format_qa_prompt(instruction: str, query: str, answer: str, hint: str):
    """ 
    Simplified Version
    """
    prompt = QAH_TEMPLATE.format(instruction=instruction, prompt=query, hint=hint, answer=answer)
    return prompt


QA_TEMPLATE = """Given the instruction: {instruction}
I need to answer the query: {query}
{condition_text}
I think: {reason}
{hint_text}
Is this reasoning correct? Provide a True/False answer as well as your rationale.
Example Answer:
Rationale: [Your rationale here]
Answer: [True/False]"""

QA_ANSWER_TEMPLATE = """Rationale: {rationale}\nAnswer: {answer}"""


NAIVE_PROMPT_TEMPLATE = """ 
Query: {query}
How would you address the query while following the instruction? Provide your thought and answer. 
Example format: 
Thought: [Your thinking process] 
Answer: [Your answer]
""" # There is issue: for roleplay scenario, we should include instruction in the system prompt and not here (!) -- Need matching evaluation here with actual inference (!)


PROPOSE_PROMPT_TEMPLATE = """
Given the instruction: {instruction}
Query: {query}
Hint: {hint}

How would you address the query while following the instruction? Provide your thought and answer. 
Example format: 
Thought: [Your thinking process] 
Answer: [Your answer]
"""

PARROT_PROMPT_TEMPLATE = """
Given the instruction: {instruction}
Query: {query}
My previous thought: {prev_thought}
My previous answer: {prev_answer}

Inspired by the previous thought and answer, how would you address the query while following the instruction? Provide your thought and answer. 
Example format: 
Thought: [Your thinking process] 
Answer: [Your answer]
"""

def prepare_naive_prompt(instruction, query):
    return NAIVE_PROMPT_TEMPLATE.format(instruction=instruction, query=query) # Matching actual inference

# Batch Inference is required for the pipeline here 
def prepare_propose_prompt(instruction, query, hint):
    return PROPOSE_PROMPT_TEMPLATE.format(instruction=instruction, query=query, hint=hint)

def prepare_parrot_prompt(instruction, query, prev_thought, prev_answer): # This one works | For hard case, do the parroting with IGP inference results | Verify it against weak human annotation
    return PARROT_PROMPT_TEMPLATE.format(instruction=instruction, query=query, prev_thought=prev_thought, prev_answer=prev_answer)

def prepare_star_prompt(instruction, query, answer):
    # Need 3 more wrong answers
    # return STAR_PROMPT_TEMPLATE.format(instruction=instruction, query=query, answer=answer)
    raise NotImplementedError


def format_qa_prompt_from_triplet(feedback, prompt, triplets):
    prompt_qa_pairs = []
    for triplet in triplets:
        condition, reason, hint = triplet
        qa_prompt = format_qa_prompt(feedback.content, prompt, condition, reason, hint)
        prompt_qa_pairs.append((prompt, qa_prompt))
    return prompt_qa_pairs


def parse_rationale_answer(response: str) -> Tuple[str, bool]:
    if "Rationale:" not in response or "Answer:" not in response:
        print("No rationale or answer found in response: \n", response)
        return "", False
    
    rationale = response.split("Rationale:")[1].split("Answer:")[0].strip()
    answer = response.split("Answer:")[1].strip().lower() == "true"
    
    # print(f"Rationale: {rationale}")
    # print(f"Answer: {answer}")
    
    return rationale, answer

def parse_thought_answer(response: str) -> Tuple[str, str]:
    
    if "Thought:" not in response or "Answer:" not in response:
        # print("No thought or answer found in response: \n", response)
        return "", ""
    
    thought = response.split("Thought:")[1].split("Answer:")[0].strip()
    answer = response.split("Answer:")[1].strip()
    return thought, answer

def parse_evaluate_answer(response: str) -> Tuple[str, str]:
    
    if "Evaluation:" not in response or "Explanation:" not in response:
        print("No evaluation or explanation found in response: \n", response)
        return "", ""
    
    evaluation_str = response.split("Evaluation:")[1].split("Explanation:")[0].strip()
    explanation = response.split("Explanation:")[1].strip()
    true_strs = ["True", "TRUE"]
    false_strs = ["False", "FALSE"]
    if evaluation_str in true_strs:
        evaluation = True
    elif evaluation_str in false_strs:
        evaluation = False
    else:
        evaluation = ""
    return evaluation, explanation


PROPOSE_RESPONSE_TEMPLATE = """Thought: {thought}
Answer: {answer}"""


EVALUATE_PROMPT_TEMPLATE = """
Given your previous answer and rationale, evaluate the following answer:

{answer}

Is this answer acceptable given the original instruction and query? Respond with True if acceptable, False if not acceptable.
Provide a brief explanation for your judgment.
Example format:
Evaluation: [Your evaluation]
Explanation: [Your explanation]
"""

ALIGNMENT_PROMPT_TEMPLATE = """
Given your previous answer and rationale, evaluate whether the following idea is correct: 

{answer} 

Is this idea aligned with your rationale? Respond with True if acceptable, False if not acceptable.
Provide a brief explanation for your judgment.
Example format:
Evaluation: [Your evaluation]
Explanation: [Your explanation]
"""

EVALUATE_RESPONSE_TEMPLATE = """Evaluation: {evaluation}
Explanation: {explanation}"""

# We should not apply chat template in the output datapoint here, since BLM & SLM process things with different template

@dataclass 
class Rationale:
    prompt: str # original prompt
    propose_prompt: str 
    correct_answer: str = "" # guide on 'how to answer' provided by Human
    parrot_prompt: str = "" # included IGP answer as hint]
    naive_prompt: str = "" # naive prompt
    thought: str = "" # thought process of the proposed answer
    answer: str = "" # proposed answer by LLM
    evaluation: Optional[bool] = None # evaluation of the correct / wrong answer 
    explanation: str = "" # explanation of the evaluation
    
    
    @property
    def is_good(self):
        is_valid = (self.evaluation is not None) and (self.explanation != "") and (self.thought != "") and (self.answer != "")
        correct_evaluation = False 
        if self.correct_answer != "": # expect True evaluation
            correct_evaluation = (self.evaluation == True)
        return is_valid and correct_evaluation
    
    @property
    def is_wrong(self):
        is_valid = (self.evaluation is not None) and (self.explanation != "") and (self.thought != "") and (self.answer != "")
        return is_valid and (self.evaluation == False)
    
    @property
    def has_answer(self):
        return self.answer != ""
            
    @property 
    def evaluate_prompt(self):
        return EVALUATE_PROMPT_TEMPLATE.format(answer=self.correct_answer)
    
    @property
    def alignment_prompt(self):
        return ALIGNMENT_PROMPT_TEMPLATE.format(answer=self.correct_answer)
    
    @property
    def evaluate_response(self):
        return EVALUATE_RESPONSE_TEMPLATE.format(evaluation=self.evaluation, explanation=self.explanation)
    
    @property
    def propose_response(self):
        return PROPOSE_RESPONSE_TEMPLATE.format(thought=self.thought, answer=self.answer)
    
    @property 
    def query_evaluate_message(self):
        return [
            {"role": "user", "content": self.propose_prompt},
            {"role": "assistant", "content": self.propose_response},
            {"role": "user", "content": self.evaluate_prompt}
        ]
        
    @property 
    def query_alignment_message(self):
        return [
            {"role": "user", "content": self.propose_prompt},
            {"role": "assistant", "content": self.propose_response},
            {"role": "user", "content": self.alignment_prompt}
        ]
    
    @property
    def instruction(self):
        return self.propose_prompt.split('\n')[0].split(':')[-1].strip()
    
    @property 
    def naive_propose_message(self):
        return [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": self.naive_prompt}
        ]
        
    def save(self, file_path):
        import json
        data = {
            "prompt": self.prompt,
            "propose_prompt": self.propose_prompt,
            "correct_answer": self.correct_answer,
            "parrot_prompt": self.parrot_prompt,
            "thought": self.thought,
            "answer": self.answer,
            "evaluation": self.evaluation,
            "explanation": self.explanation
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
       
    @classmethod 
    def load(cls, file_path):
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
        
    @property
    def message(self):
        assert self.evaluation is not None, "Evaluation is not completed yet"
        return [
            {"role": "user", "content": self.propose_prompt},
            {"role": "assistant", "content": self.propose_response},
            {"role": "user", "content": self.evaluate_prompt},
            {"role": "assistant", "content": self.evaluate_response}
        ]
   
class IGReasoner: 
    
    def __init__(self, get_response, rationales, apply_chat_template=lambda x: x, n_answer_per_question=3):
        self.get_response = get_response
        self.apply_chat_template = apply_chat_template
        self.n_answer_per_question = n_answer_per_question
        self.rationales = rationales

    @classmethod 
    def make_from_tuples(cls, get_response, prompt_tuples=[], apply_chat_template=lambda x: x, n_answer_per_question=3):
        rationales = []
        for _ in range(n_answer_per_question):
            for (prompt, propose_prompt, correct_answer, parrot_prompt, naive_prompt) in prompt_tuples:
                rationales.append(Rationale(prompt, propose_prompt, correct_answer, parrot_prompt, naive_prompt))
                
        return cls(get_response=get_response, rationales=rationales, apply_chat_template=apply_chat_template, n_answer_per_question=n_answer_per_question)
        
    @classmethod
    def make_with_augmentation(cls, get_response, rationales, apply_chat_template=lambda x: x, multiplication_factor: int = 20):
        unsolved_cases = sum(1 for rationale in rationales if not rationale.is_good)
        print(f"Currently, we have {unsolved_cases} unsolved cases. Augment each unsolved case by {multiplication_factor} times and initialize BLM reasoner to work on {unsolved_cases * multiplication_factor} hard cases.")
        new_rationales = []
        for rationale in rationales:
            if not rationale.is_good:
                for _ in range(multiplication_factor):
                    new_rationales.append(Rationale(
                        prompt=rationale.prompt,
                        propose_prompt=rationale.propose_prompt,
                        correct_answer=rationale.correct_answer,
                        parrot_prompt=rationale.parrot_prompt,
                        naive_prompt=rationale.naive_prompt,
                        thought=rationale.thought,
                        answer=rationale.answer,
                        evaluation=rationale.evaluation,
                        explanation=rationale.explanation
                    ))
            else:
                new_rationales.append(rationale)
        
        new_instance = cls(get_response, new_rationales, apply_chat_template, int(multiplication_factor * 2))
        return new_instance    
    
    @classmethod 
    def make_for_mistakes(cls, get_response, rationales, apply_chat_template=lambda x: x, multiplication_factor: int = 20):
        correct_cases = sum([1 for rationale in rationales if rationale.is_good])
        print(f"Currently, we have {correct_cases} correct cases. Augment each correct case by {multiplication_factor} times and initialize BLM reasoner to work on {correct_cases * multiplication_factor} hard cases.")
        new_rationales = []
        for rationale in rationales:
            if rationale.is_good:
                for _ in range(multiplication_factor):
                    new_rationales.append(Rationale(
                        prompt=rationale.prompt,
                        propose_prompt=rationale.propose_prompt,
                        correct_answer=rationale.correct_answer,
                        parrot_prompt=rationale.parrot_prompt,
                        naive_prompt=rationale.naive_prompt,
                        thought=rationale.thought,
                        answer=rationale.answer,
                        evaluation=rationale.evaluation,
                        explanation=rationale.explanation
                    ))
            else:
                new_rationales.append(rationale)
        
        new_instance = cls(get_response, new_rationales, apply_chat_template, int(multiplication_factor * 2))
        return new_instance
                
    
    
    @classmethod
    def load_from_rationales(cls, get_response, rationale_dir, apply_chat_template=lambda x: x, multiplication_factor: int = 20):
        rationales = []
        files = glob.glob(rationale_dir + "/*")
        for file_path in files:
            rationales.append(Rationale.load(file_path))
        print("Loaded {} rationales".format(len(rationales)))
        return cls.make_with_augmentation(get_response, rationales, apply_chat_template, multiplication_factor)
    
    @classmethod 
    def load_from_star_rationales(cls, get_response, star_rationale_dir, apply_chat_template=lambda x: x, multiplication_factor: int = 20):
        rationales = []
        files = glob.glob(star_rationale_dir + "/*")
        for file_path in files:
            rationales.append(Rationale.load(file_path))
        return cls.make_for_mistakes(get_response, rationales, apply_chat_template, multiplication_factor)
    
    def save_rationales(self, rationale_dir):
        os.makedirs(rationale_dir, exist_ok=True)
        for idx, rationale in enumerate(self.rationales):
            file_path = os.path.join(rationale_dir, f"rationale_{idx}.json")
            rationale.save(file_path)
        
        print(f"Saved {len(self.rationales)} rationales to {rationale_dir}")
                
    def format_evaluate_prompt(self, rationale: Rationale):
        completion = "####"
        msg = rationale.query_evaluate_message + [{"role": "assistant", "content": completion}]
        tmp = self.apply_chat_template(msg)
        if isinstance(tmp, list): # Case with API calls
            return rationale.query_evaluate_message
        else:
            return tmp.split(completion)[0] # query prompt only for vLLM 
        
    def format_naive_prompt(self, rationale: Rationale):
        completion = "####"
        msg = rationale.naive_propose_message + [{"role": "assistant", "content": completion}]
        tmp = self.apply_chat_template(msg)
        if isinstance(tmp, list): # Case with API calls
            return rationale.naive_propose_message
        else:
            return tmp.split(completion)[0] # query prompt only for vLLM 
        
    def format_alignment_prompt(self, rationale: Rationale):
        completion = "####"
        msg = rationale.query_alignment_message + [{"role": "assistant", "content": completion}]
        tmp = self.apply_chat_template(msg)
        if isinstance(tmp, list): # Case with API calls
            return rationale.query_alignment_message
        else:
            return tmp.split(completion)[0] # query prompt only for vLLM inference

    @property
    def unanswered_indices(self):
        return [i for i, rationale in enumerate(self.rationales) if rationale.evaluation is None]
    
    @property
    def failed_indices(self):
        # TODO: Inclusion of False Answer requires changing on this function as well
        return [i for i, rationale in enumerate(self.rationales) if not rationale.is_good]
    
    def eval(self, check_alignment=True, batch_size=5000):
        
        indices = self.failed_indices
        print(f"-- Evaluating {len(indices)} cases ...")
        
        prompts = [self.format_naive_prompt(rationale) for rationale in self.rationales if self.rationales.index(rationale) in indices]
        
        responses = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            batch_responses = self.get_response(batch)
            responses.extend(batch_responses)
            
        for i, response in zip(indices, responses):
            thought, answer = parse_thought_answer(response)
            self.rationales[i].thought = thought
            self.rationales[i].answer = answer
        
        if check_alignment:
            prompts = [self.format_alignment_prompt(rationale) for rationale in self.rationales if self.rationales.index(rationale) in indices]
            
            responses = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i+batch_size]
                batch_responses = self.get_response(batch)
                responses.extend(batch_responses)
                
            for i, response in zip(indices, responses):
                evaluation, explanation = parse_evaluate_answer(response)
                self.rationales[i].evaluation = evaluation
                self.rationales[i].explanation = explanation
        else:
            print("Direct evaluation for GSM8K ...")
            for i, rationale in enumerate(self.rationales):
                evaluation = rationale.correct_answer.split("Answer: ")[-1].strip() in rationale.answer
                self.rationales[i].evaluation = evaluation
                self.rationales[i].explanation = "Aligned with correct answer"
            
        # Analysis on how many cases solved & unsolved
        solved_cases = sum(1 for rationale in self.rationales if rationale.is_good)        
        success_rate = solved_cases / len(self.rationales) * 100 
        
        print(f"Evaluation Result: {success_rate:.2f}%")
        
        # Store CSV file 
        data = []
        for rationale in self.rationales:
            data_dict = {
                "prompt": rationale.prompt,
                "answer": rationale.answer,
                "evaluation": rationale.evaluation,
                "correct_answer": rationale.correct_answer
            }
            data.append(data_dict)
            
        import pandas as pd 
        df_anal = pd.DataFrame(data)
        return df_anal 
    
    def think(self, include_failed=True, do_parrot=False, do_naive=False, check_alignment=True, batch_size=5000):
        
        indices = self.failed_indices if include_failed else self.unanswered_indices
        print(f"-- Dealing with {len(indices)} cases ...")
    
        # Propose 
        if do_parrot: # For Hard cases, do the parroting
            prompts = [rationale.parrot_prompt for rationale in self.rationales if self.rationales.index(rationale) in indices]
            responses = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i+batch_size]
                batch_responses = self.get_response(batch)
                responses.extend(batch_responses)
        elif not do_naive: # For not-so-hard cases, do hinted generation
            prompts = [rationale.propose_prompt for rationale in self.rationales if self.rationales.index(rationale) in indices]
            responses = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i+batch_size]
                batch_responses = self.get_response(batch)
                responses.extend(batch_responses)
        else:
            prompts = [self.format_naive_prompt(rationale) for rationale in self.rationales if self.rationales.index(rationale) in indices]
            responses = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i+batch_size]
                batch_responses = self.get_response(batch)
                responses.extend(batch_responses)
        
        for i, response in zip(indices, responses):
            thought, answer = parse_thought_answer(response)
            self.rationales[i].thought = thought
            self.rationales[i].answer = answer
        
        # Evaluate
        if check_alignment:
            responses = self.get_response([self.format_alignment_prompt(rationale) for rationale in self.rationales if self.rationales.index(rationale) in indices])
            for i, response in zip(indices, responses):
                evaluation, explanation = parse_evaluate_answer(response)
                self.rationales[i].evaluation = evaluation
                self.rationales[i].explanation = explanation
        else: # Direct evaluation (for GSM8K specifically)
            print("Direct evaluation for GSM8K ...")
            for i, rationale in enumerate(self.rationales):
                evaluation = rationale.correct_answer.split("Answer: ")[-1].strip() in rationale.answer
                self.rationales[i].evaluation = evaluation
                self.rationales[i].explanation = "Aligned with correct answer"
   
        # Analysis on how many cases solved & unsolved
        solved_cases = sum(1 for rationale in self.rationales if rationale.is_good)
        unsolved_cases = len(self.rationales) - solved_cases
        
        prev_unsolved_count = len(indices)
        curr_unsolved_count = unsolved_cases
        print(f"Thinking process solved {prev_unsolved_count - curr_unsolved_count} cases, {curr_unsolved_count} cases remaining unsolved.")
        
        
    def make_mistakes(self, batch_size=5000):
        """
        Searching for mistakes 
        """
        indices = [i for i, rationale in enumerate(self.rationales) if rationale.is_good or rationale.evaluation is None]
        print(f"Expecting mistakes on {len(indices)} more cases ...")
        
        prompts = [rationale.propose_prompt for rationale in self.rationales if self.rationales.index(rationale) in indices]
        responses = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            batch_responses = self.get_response(batch)
            responses.extend(batch_responses)
            
        for i, response in zip(indices, responses):
            thought, answer = parse_thought_answer(response)
            self.rationales[i].thought = thought
            self.rationales[i].answer = answer
        
        responses = self.get_response([self.format_alignment_prompt(rationale) for rationale in self.rationales if self.rationales.index(rationale) in indices])

        for i, response in zip(indices, responses):
            evaluation, explanation = parse_evaluate_answer(response)
            self.rationales[i].evaluation = evaluation
            self.rationales[i].explanation = explanation
        
        # Statistics
        solved_cases = sum(1 for rationale in self.rationales if not rationale.is_good)
        unsolved_cases = len(self.rationales) - solved_cases
        
        prev_unsolved_count = len(indices)
        curr_unsolved_count = unsolved_cases
        print(f"Mistake-making process messed up {prev_unsolved_count - curr_unsolved_count} cases, {curr_unsolved_count} cases remaining unsolved.")        
        
        
    def igr_data(self, prompt):
        """ 
        - We will output a message object for now | post-processing could be carried out in the feedback pipeline        
        Return correct rationales on association regarding a specific prompt
        """
        data = []
        for rationale in self.rationales:
            if rationale.prompt == prompt and rationale.is_good: # Good rationale gets slotted into the igr_data only
                data.append(rationale.message)
        return data
    
    def issue_prompts(self): # Align Name with Inference Stage
        return [rationale.prompt for rationale in self.rationales if (not rationale.is_good)]
    
    def is_weak(self, prompt):
        return prompt in self.issue_prompts()
    
    
@dataclass 
class Choice:
    prompt: str # original prompt
    choice_prompt: str 
    correct_answer: str = "" # guide on 'how to answer' provided by Human
    explain_prompt: str = "" # included IGP answer as hint]
    thought: str = "" # thought process of the proposed answer
    answer: str = "" # proposed answer by LLM
    
    
    @property 
    def is_valid(self):
        return (self.thought and self.answer and self.correct_answer and self.choice_prompt and self.prompt)
    
    @property
    def is_good(self):
        is_correct = (self.answer.upper() == self.correct_answer.upper())
        return is_correct and self.is_valid
    
    def message(self, instruction):
        answer_prompt = f"""Thought: {self.thought}\nAnswer: {self.answer}"""
        return [{"role": "system", "content": instruction},
               {"role": "user", "content": self.choice_prompt},
               {"role": "assistant", "content": answer_prompt}]
    
    def save(self, file_path):
        import json
        data = {
            "prompt": self.prompt,
            "choice_prompt": self.choice_prompt,
            "correct_answer": self.correct_answer,
            "explain_prompt": self.explain_prompt,
            "thought": self.thought,
            "answer": self.answer
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
       
    @classmethod 
    def load(cls, file_path):
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    
class StarReasoner: 
    
    def __init__(self, instruction, get_response, choices, apply_chat_template=lambda x: x, n_answer_per_question=1):
        self.instruction = instruction
        self.get_response = get_response
        self.apply_chat_template = apply_chat_template
        self.n_answer_per_question = n_answer_per_question
        self.choices = choices

    @classmethod 
    def make_from_tuples(cls, instruction, get_response, prompt_tuples=[], apply_chat_template=lambda x: x, n_answer_per_question=1):
        print("Making STaR thinker ...")
        choices = []
        for _ in range(n_answer_per_question):
            for (prompt, star_prompt, correct_letter, explain_choice_prompt) in prompt_tuples:
                choices.append(Choice(prompt, star_prompt, correct_letter, explain_choice_prompt))
                
        return cls(instruction=instruction, get_response=get_response, choices=choices, apply_chat_template=apply_chat_template, n_answer_per_question=n_answer_per_question)
    
    @classmethod 
    def load_from_choices(cls, get_response, star_choice_dir, apply_chat_template=lambda x: x, multiplication_factor: int = 20):
        choices = []
        files = glob.glob(star_choice_dir + "/*")
        for file_path in files:
            choices.append(Choice.load(file_path))
        return cls.make_for_mistakes(get_response, choices, apply_chat_template, multiplication_factor)
    
    def save_choices(self, choice_dir):
        os.makedirs(choice_dir, exist_ok=True)
        for idx, choice in enumerate(self.choices):
            file_path = os.path.join(choice_dir, f"choice_{idx}.json")
            choice.save(file_path)
        
        print(f"Saved {len(self.choices)} choices to {choice_dir}")
                
                
    def format_choice_prompt(self, choice: Choice):
        completion = "####"
        msg = [{"role": "system", "content": self.instruction},
               {"role": "user", "content": choice.choice_prompt},
               {"role": "assistant", "content": completion}]
        return self.apply_chat_template(msg).split(completion)[0]
    
    def format_explain_prompt(self, choice: Choice):
        completion = "####"
        msg = [{"role": "system", "content": self.instruction},
               {"role": "user", "content": choice.explain_prompt},
               {"role": "assistant", "content": completion}]
        return self.apply_chat_template(msg).split(completion)[0]

    @property
    def failed_indices(self):
        # TODO: Inclusion of False Answer requires changing on this function as well
        return [i for i, choice in enumerate(self.choices) if not choice.is_good]
    
    def star_data(self, prompt):
        return [choice.message(self.instruction) for choice in self.choices if choice.prompt == prompt]
    
    def think(self, do_choice=True, batch_size=5000):
        
        indices = self.failed_indices
        print(f"-- Dealing with {len(indices)} cases ...")
    
        # Propose 
        if do_choice: # For Hard cases, do the parroting
            prompts = [choice.choice_prompt for choice in self.choices if self.choices.index(choice) in indices]
            responses = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i+batch_size]
                batch_responses = self.get_response(batch)
                responses.extend(batch_responses)
            print("Choice process finished.")
            
            for i, response in zip(indices, responses):
                thought, answer = parse_thought_answer(response)
                self.choices[i].thought = thought
                self.choices[i].answer = answer
            print("Choices parsing complete.")    
        else:
            prompts = [choice.explain_prompt for choice in self.choices if self.choices.index(choice) in indices]
            responses = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i+batch_size]
                batch_responses = self.get_response(batch)
                responses.extend(batch_responses)
                
            for i, response in zip(indices, responses):
                thought, answer = parse_thought_answer(response)
                self.choices[i].thought = thought
                self.choices[i].answer = self.choices[i].correct_answer
                
        print("Doing statistics ...")
        # Statistics
        solved_cases = sum(1 for choice in self.choices if choice.is_good)
        unsolved_cases = len(self.choices) - solved_cases
        print(f"Thinking process solved {solved_cases} cases, {unsolved_cases} cases remaining unsolved.")