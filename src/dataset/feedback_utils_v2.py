import re
import os
import json
from enum import Enum
from uuid import uuid5, UUID
from typing import Optional, Any, Callable
import datasets
import networkx as nx
from tqdm import tqdm
import numpy as np
from typing import Tuple, Callable
from datasets import Dataset
from transformers import AutoTokenizer
from .prompts_v2 import GENERATE_PROMPT_TEMPLATE, parse_prompt_from_response

# Used to generate deterministic UUIDs for feedback
NAMESPACE_UUID = UUID("00000000-0000-0000-0000-000000000000")

def sample_prompts(content: str, get_response: Callable, idea: str = ""):
    """ 
    Sample prompts from the content
    """
    generate_prompt = GENERATE_PROMPT_TEMPLATE.format(content=content, Idea=idea)
    response = get_response(generate_prompt)
    prompts = parse_prompt_from_response(response)
    return prompts

def process_query(query):
    processed_query = query 
    processed_query = ("\n").join([line for line in processed_query.split("\n") if "Hint:" not in line])
    return processed_query

def load_graph_and_image(feedback, query):
    """ 
    Load Graph & Image from Query
    """
    img_folder = f"database/{feedback.file_name}/images"
    img_name = query.replace(" ", "-") + ".png"
    net_name = query.replace(" ", "-") + ".json"

    # Read the image
    image_path = os.path.join(img_folder, img_name)
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
    else:
        print(f"Image file not found at {image_path}")

    # Read the networkx graph
    graph_path = os.path.join(img_folder, net_name)
    if os.path.exists(graph_path):
        with open(graph_path, "r") as graph_file:
            graph_data = json.load(graph_file)
        graph = nx.node_link_graph(graph_data)
    else:
        print(f"Graph file not found at {graph_path}")

    return img_data, graph 

# Ok so here is where the real deal happens
def make_ft_dataset(igr_dataset, tokenizer):
    ft_dataset = []
    for message in igr_dataset: 
        assert isinstance(message, list), "Message must be a list"
        # Do a split of the message, and collect (prompt, completion) each time we see an 'assistant' response 
        for idx in range(len(message)):
            if message[idx]['role'] == 'assistant':
                completion_txt = message[idx]["content"].strip() # Special gadget required for chat_template | it strip in the backend
                prompt = tokenizer.apply_chat_template(message[:idx+1], tokenize=False)
                query_prompt, completion_prompt = prompt.split(completion_txt)
                ft_dataset.append({"prompt": query_prompt, "completion": completion_txt+completion_prompt})
    # Convert to Huggingface Dataset object
    ft_dataset = Dataset.from_list(ft_dataset)
    ft_dataset = {"train": ft_dataset}
    return ft_dataset


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

# def make_ft_dataset(igr_dataset, tokenizer):
#     ft_dataset = []
#     for query, response in igr_dataset:
#         msg = [{"role": "user", "content": query}, {"role": "assistant", "content": response}]
#         full_prompt = tokenizer.apply_chat_template(msg, tokenize=False)
#         prompt = full_prompt.split(response)[0]
#         completion = response + full_prompt.split(response)[1]
#         ft_dataset.append({"prompt": prompt, "completion": completion})
#     ft_dataset = Dataset.from_list(ft_dataset)
#     return {"train": ft_dataset}

# First Principle: ICL response is already good enough | If ICL is good enough, it's all about prompt-completion and error cases
# Self-supervision loss based on a verbal feedback: Loss(Prompted response, FineTuned Response)
# -- Question is how to compress the prompt into the model --> REFT
# -- Question is how to compress REFT into the mode --> Fine-Tuning 

def mix_and_mingle(igr_dataset, ratio = [0.44, 0.56]):
    """ 
    Collect Weak & Strong Associations 
    Mix according to specified proportion
    """
    # Conduct statistical analysis on igr_dataset
    total_samples = len(igr_dataset)
    weak_samples = sum(1 for _, _, is_weak in igr_dataset if is_weak)
    strong_samples = total_samples - weak_samples

    print(f"Total samples: {total_samples}")
    print(f"Weak samples: {weak_samples}")
    print(f"Strong samples: {strong_samples}")
    print(f"Percentage of weak samples: {weak_samples / total_samples * 100:.2f}%")
    # Mix and Mingle on dataset to ensure weak & strong association is according to the pre-set ratio
    # Calculate the numbers if we keep all weak associations
    if weak_samples <= total_samples * ratio[0]:
        weak_keep = weak_samples
        strong_keep = min(strong_samples, int(total_samples * ratio[1]))
    else:
        # Calculate the numbers if we keep all strong associations
        strong_keep = strong_samples
        weak_keep = min(weak_samples, int(total_samples * ratio[0]))

    # Ensure we're not exceeding available samples
    weak_keep = min(weak_keep, weak_samples)
    strong_keep = min(strong_keep, strong_samples)

    # Separate weak and strong samples
    weak_data = [item[:2] for item in igr_dataset if item[2]]
    strong_data = [item[:2] for item in igr_dataset if not item[2]]

    # Randomly sub-sample the data

    # Convert weak_data to a numpy array of indices
    weak_indices = np.arange(len(weak_data))
    chosen_weak_indices = np.random.choice(weak_indices, weak_keep, replace=False)
    weak_data = [weak_data[i] for i in chosen_weak_indices]

    # Convert strong_data to a numpy array of indices
    strong_indices = np.arange(len(strong_data))
    chosen_strong_indices = np.random.choice(strong_indices, strong_keep, replace=False)
    strong_data = [strong_data[i] for i in chosen_strong_indices]

    # Combine and shuffle the dataset
    igr_dataset = weak_data + strong_data
    
    print("----- Proportion after Mixing -----")
    print(f"Weak samples: {len(weak_data)}")
    print(f"Strong samples: {len(strong_data)}")
    print(f"Percentage of weak samples: {len(weak_data) / (len(weak_data) + len(strong_data)) * 100:.2f}%")
    
    np.random.shuffle(igr_dataset)
    return igr_dataset


EVALUATE_PROMPT_TEMPLATE = """
Given the instruction: {instruction}
Query: {query}
Hint: {hint}
Response: {response}

Evaluate if the response follows the instruction and appropriately addresses the query. Use the hint as a guide for the correct answer. Determine if the response is acceptable based on these criteria.

Please provide your evaluation in the following format:
Evaluation: [True/False]
Explanation: [Brief explanation for your judgment]
"""

from datasets import load_dataset

class Feedback:
    content: str
    prompts: list # Places where feedback apply
    search_infos: dict # Search Information
    weak_anno: bool
    num_train: int = 200 # Number of training samples

    def __init__(self, content: str, weak_anno: bool = True):
        self.content = content
        self.prompts = []
        self.correct_responses = {}
        self.search_infos = {}
        self.weak_anno = weak_anno
        
        if content == "Provide your thought and answer to the question from user. For example:\n        Thought: ...\n        Answer: ...":
            print("Loading GSM8K dataset ...")
            gsm_ds = load_dataset("Ksgk-fy/gsm8k")
            train_prompts = gsm_ds["train"]["user"]
            test_prompts = gsm_ds["test"]["user"]
            self.prompts = train_prompts + test_prompts
            self.map_prompt_to_str = {prompt: prompt[:100].replace(" ", "-") for prompt in self.prompts}
            self.map_prompt_to_str = {prompt: prompt[:100].replace("/", "") for prompt in self.prompts}
            self.num_train = len(train_prompts)
            
            answers = gsm_ds["train"]["assistant"]
            annotations = [{"query": prompt, "weak_anno": answer, "strong_anno": answer} for prompt, answer in zip(self.prompts, answers)]
            self.correct_responses = {}
            for info in annotations:
                prompt = info["query"]
                prompt_str = self.map_prompt_to_str[prompt]
                if self.weak_anno:
                    self.correct_responses[prompt_str] = info["weak_anno"]
                else:
                    self.correct_responses[prompt_str] = info["strong_anno"]
        else:
            self.map_prompt_to_str = {prompt: prompt.replace(" ", "-") for prompt in self.prompts}
        try:
            self.load_info()
            print("Loaded {} training prompts".format(len(self.prompts[:self.num_train])))
            print("Loaded {} test prompts".format(len(self.prompts[self.num_train:])))
            print("Loaded {} annotations".format(len(self.correct_responses)))
        except:
            print("Completion Information not found.")
            
    @property
    def test_prompts(self):
        return self.prompts[self.num_train:]

    @property
    def id(self):
        return uuid5(NAMESPACE_UUID, self.content)

    
    @property
    def file_name(self):
        assert self.id is not None, "Feedback must have an ID to have a file name"
        content = self.content.lower()[:30]
        content = re.sub(r"[^a-z0-9 ]", " ", content)
        content = re.sub(r" +", " ", content)
        content = content.replace(" ", "_")
        content = content.strip()
        return f"{content}_{self.id}"
    
    def sample_prompts(self, get_response: Callable, idea: str = ""):
        sampled_prompts = sample_prompts(self.content, get_response, idea)
        self.prompts.extend(sampled_prompts)
        print("Sampled {} prompts".format(len(sampled_prompts)))
    
    def load_info(self):

        if not self.correct_responses:
            try:
                for info in self.annotations:
                    prompt = info["query"]
                    prompt_str = self.map_prompt_to_str[prompt]
                    if self.weak_anno:
                        self.correct_responses[prompt_str] = info["weak_anno"]
                    else:
                        self.correct_responses[prompt_str] = info["strong_anno"]
            except: 
                print("Annotations not found")
        
        if not self.prompts:
            print("Loading prompts ...")
            try:
                with open(f"database/{self.file_name}/prompts.json", "r") as f:
                    prompts = json.load(f)
                if prompts:
                    self.prompts = prompts 
            except:
                print("Prompts not found")
        return

    def save_info(self):
        os.makedirs(f"database/{self.file_name}", exist_ok=True)
        
        with open(f"database/{self.file_name}/prompts.json", "w") as f:
            json.dump(self.prompts, f)
        return
    
    def save_prompts(self):
        os.makedirs(f"database/{self.file_name}", exist_ok=True)
        
        with open(f"database/{self.file_name}/prompts.json", "w") as f:
            json.dump(self.prompts, f)
            
    def load_graph_and_image(self, query):
        return load_graph_and_image(self, query)
    
    def load_response(self, query, mode: str = "", model: str = "claude"):
        file_path = f"database/{self.file_name}/{mode}_responses_{model}.json"
        with open(file_path, "r") as f:
            responses = json.load(f)
        try:
            response_key = query.replace(" ", "-")
            response = responses[response_key]
        except:
            query_idx = None 
            for idx, prompt in enumerate(self.prompts):
                if prompt == query:
                    query_idx = idx 
            response = responses[query_idx]
            
        response = response.split("Issue response:")[-1]
        return response
    
    @property
    def annotations(self):
        with open(f"database/{self.file_name}/annotations.json", "r") as f:
            annotation = json.load(f)
        return annotation
    
    def save_annotation(self, annotation):
        with open(f"database/{self.file_name}/annotations.json", "w") as f:
            json.dump(annotation, f)
            
    def annotate(self, mode="IGP"):
        annotations = []
        for prompt in tqdm(self.prompts, desc="Annotating How to answer the query"):
            print("----- Rule: \n", self.content)
            print("----- Query: \n", prompt)
            print("----- How to answer the query: ")
            # weak_anno = input()
            weak_anno = ""
            anno_dict = {"weak_anno": weak_anno, "prompt": prompt}
            annotations.append(anno_dict)
        self.save_annotation(annotations)
        
    def evaluate_alignment(self, mode, model, get_response: Callable):
        evaluations = []
        explanations = []
        for (prompt, annotation) in tqdm(zip(self.prompts, self.annotations), desc="Evaluating alignment"):
            response = self.load_response(prompt, mode, model)
            evaluate_prompt = EVALUATE_PROMPT_TEMPLATE.format(instruction=self.content, query=prompt, hint=annotation, response=response)
            response = get_response(evaluate_prompt)
            evaluation, explanation = parse_evaluate_answer(response)
            evaluations.append(evaluation)
            explanations.append(explanation)
        return evaluations, explanations

    def update_feedback_search_completion(self):
        search_infos = {}
        for prompt in self.prompts:
            # Get completion file -- There are bunch of rejected completions, and accepted completions
            get_prompt_complete_id = lambda prompt: "search_info_"+prompt.replace(" ","-").replace(".","")
            file_name = get_prompt_complete_id(prompt)
            file_path = f"database/{self.file_name}/{file_name}.json"
            with open(file_path, "r") as f:
                search_info = json.load(f)
            search_infos[prompt] = search_info
            os.remove(file_path) # Remove File
        self.search_infos = search_infos

    def boostrap_augment(self, aug_name="aug_r1"):
        """ 
        Concatenate Extra augmented prompts into the feedback object
        """
        try:
            dest_dir = f'database/{self.file_name}/{aug_name}'
            import glob, json
            aug_prompts = []
            aug_search_infos = {}
            for file in glob.glob(dest_dir + "/*.json"):
                with open(file, "r") as f:
                    data = json.load(f)
                prompt = data[0]['prompt']
                prompt_id = file.split(".json")[0].split("search_info_")[-1]
                aug_search_infos[prompt_id] = data
                aug_prompts.append(prompt)

            self.prompts = self.prompts + aug_prompts
            self.search_infos.update(aug_search_infos) # Update the search info as well
            print("Augmenting {} prompts".format(len(aug_prompts)))
            print("Database Updated!")
        except:
            print(f"Augmentation file {aug_name} not found.")
            
            
    def save_IGR_dataset(self, prompt: str, igr_dataset: dict, iter_n: int = 1, ID: str = ""):
        """ 
        save IGR dataset to the file
        iter_n: current iteration number
        """
        os.makedirs(f'database/{self.file_name}/IGR{ID}_learn_iter{iter_n}', exist_ok=True)
        prompt_str = self.map_prompt_to_str[prompt]
        filename = f'database/{self.file_name}/IGR{ID}_learn_iter{iter_n}/{prompt_str}.json'
        with open(filename, 'w') as f:
            json.dump(igr_dataset, f, indent=2)        
            
            
    def load_IGR_dataset(self, iter_n: int = 1, ID: str = "", train=True):
        """ 
        Get IGR learning dataset
        iter_n: The current iteration number
        """
        
        igr_dataset = []
        for iter_num in range(1, iter_n + 1):
            if train:
                prompts = self.prompts[:self.num_train]
            else:
                prompts = self.prompts[self.num_train:]
            for prompt in tqdm(prompts):
                prompt_str = self.map_prompt_to_str[prompt]

                igr_file = f"database/{self.file_name}/IGR{ID}_learn_iter{iter_num}/{prompt_str}.json"

                if os.path.exists(igr_file):
                    try:
                        with open(igr_file, "r") as f:
                            igr_data = json.load(f)
                        igr_dataset.extend(igr_data)
                        
                    except Exception as e:
                        print(f"Error when loading prompt: {prompt} for iteration {iter_num}")
                        print(f"Error details: {str(e)}")
                        continue
                else:
                    print(f"IGR file not found for prompt: {prompt} for iteration {iter_num}")
                    
        print("Collected {} samples".format(len(igr_dataset)))
        
        return igr_dataset 
    
    def save_STaR_dataset(self, prompt: str, star_dataset: dict, iter_n: int = 1, ID: str = ""):
        """ 
        save STaR dataset to the file
        iter_n: current iteration number
        """
        os.makedirs(f'database/{self.file_name}/STaR{ID}_learn_iter{iter_n}', exist_ok=True)
        prompt_str = self.map_prompt_to_str[prompt]
        filename = f'database/{self.file_name}/STaR{ID}_learn_iter{iter_n}/{prompt_str}.json'
        with open(filename, 'w') as f:
            json.dump(star_dataset, f, indent=2)
    
    def load_STaR_dataset(self, iter_n: int = 1, ID: str = "", train=True):
        """ 
        Get STaR learning dataset
        iter_n: The current iteration number
        """
        star_dataset = []
        for iter_num in range(1, iter_n + 1):
            if train:
                prompts = self.prompts[:self.num_train]
            else:
                prompts = self.prompts[self.num_train:]
                
            for prompt in tqdm(prompts):
                prompt_str = self.map_prompt_to_str[prompt]

                star_file = f"database/{self.file_name}/STaR{ID}_learn_iter{iter_num}/{prompt_str}.json"

                if os.path.exists(star_file):
                    try:
                        with open(star_file, "r") as f:
                            igr_data = json.load(f)
                        star_dataset.extend(igr_data)
                        
                    except Exception as e:
                        print(f"Error when loading prompt: {prompt} for iteration {iter_num}")
                        print(f"Error details: {str(e)}")
                        continue
                else:
                    print(f"STaR file not found for prompt: {prompt} for iteration {iter_num}")
                    
        print("Collected {} samples".format(len(star_dataset)))
        return star_dataset 

    def prepare_IGR_dataset(self, tokenizer: AutoTokenizer, iter_n: int = 1, ID: str = "", train=True):
        igr_dataset = self.load_IGR_dataset(iter_n, ID, train)
        return make_ft_dataset(igr_dataset, tokenizer)
    
    def prepare_SFT_dataset(self, tokenizer: AutoTokenizer, train=True):
        sft_dataset = []
        if train:
            prompts = self.prompts[:self.num_train]
        else:
            prompts = self.prompts[self.num_train:]
            
        for prompt in prompts:
            igp_response = self.load_response(prompt, mode="IGP")
            msg = [
                {"role": "system", "content": self.content}, 
                {"role": "user", "content": prompt}, 
                {"role": "assistant", "content": igp_response}
            ]
            sft_dataset.append(msg)
        return make_ft_dataset(sft_dataset, tokenizer)
    
    def prepare_STaR_dataset(self, tokenizer: AutoTokenizer, iter_n: int = 1, ID: str = "", train=True):
        star_dataset = self.load_STaR_dataset(iter_n, ID, train)
        return make_ft_dataset(star_dataset, tokenizer)