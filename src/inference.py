####################################################
######## Inference with PEFT  Checkpoint     #######
####################################################

import torch
import transformers
from datasets import load_dataset
from peft import PeftModel
from peft import PeftConfig
import pandas as pd
from tqdm import tqdm as tqdm 
import json 
from transformers import HfArgumentParser
from .utils import ModelArguments


device = "cuda" if torch.cuda.is_available() else "cpu"


def formatting_query_prompt_func(prompt, tokenizer,
                                 completion="####Dummy-Answer"):
    messages=[
        {"role": "user","content": prompt,},
        {"role": "assistant","content": completion,}
    ]
    format_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    query_prompt = format_prompt.split(completion)[0]
    return query_prompt

def formattting_query_prompt_func_with_sys(prompt, sys_prompt,
                                           tokenizer,
                                           completion = "####Dummy-Answer"):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion}
    ]
    format_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    query_prompt = format_prompt.split(completion)[0]
    return query_prompt


class PeftInferencer:
    def __init__(self,adaptor_id, use_quant=False):
        # self.base_model_id = base_model_id
        self.adaptor_id = adaptor_id
        self.model_max_length = 4096
        self.load_model(use_quant=use_quant)
        self.load_peft_adaptor()
    
    def load_model(self, use_quant=False):
        config = PeftConfig.from_pretrained(self.adaptor_id)
        self.base_model_id = config.base_model_name_or_path

        if use_quant:
            # BitsAndBytesConfig int-4 config
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None
            
        # Load model and tokenizer
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.base_model_id, 
            model_max_length=self.model_max_length, 
            padding_side="right", 
            use_fast=False)

        if "Meta-Llama-3-" in self.base_model_id:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.terminators = self.tokenizer.eos_token_id
        
    def load_peft_adaptor(self):
        model = PeftModel.from_pretrained(self.model, self.adaptor_id)
        del self.model
        self.model = model

    def generate(self, prompt, sys_prompt = ""):
        """ 
        Generate response from PeFT model: during training, no system prompt is placed in the input
        """
        if sys_prompt == "":
            query_prompt = formatting_query_prompt_func(prompt, self.tokenizer)
        else:
            query_prompt = formattting_query_prompt_func_with_sys(prompt, sys_prompt, self.tokenizer)
        inputs = self.tokenizer(query_prompt, return_tensors="pt").to(device)
        response = self.model.generate(**inputs, max_new_tokens=512, do_sample=True, eos_token_id=self.terminators, early_stopping=True)
        response = self.tokenizer.decode(response[0])
        return response

    def batch_generate(self, prompts, batch_size=8):
        """
        Generate responses for a batch of prompts using the PeFT model.
        This method performs batch inference to speed up the generation process.
        """
        # Slice the prompts according to batch_size
        batched_prompts = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]

        all_responses = []
        pb  = tqdm(total=len(batched_prompts), desc="Generating responses in batch")
        for batch in batched_prompts:
            # Tokenize the batch of prompts
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Generate responses in batch
            responses = self.model.generate(**inputs, max_new_tokens=512, do_sample=True, eos_token_id=self.terminators, early_stopping=True)
            
            # Decode the responses
            decoded_responses = [self.tokenizer.decode(response, skip_special_tokens=True) for response in responses]
            all_responses.extend(decoded_responses)
            pb.update(1)
        
        return all_responses
    
    def merge_and_upload(self, model_id):
        """ 
        Merge adaptor with base model & upload to HF repository 
        """
        merge_model = self.model.merge_and_unload()
        merge_model.push_to_hub(model_id) # Merge and Push to Huggingface
        self.tokenizer.push_to_hub(model_id)
        print(f"Model and tokenizer are merged and uploaded to {model_id}")
        
    
def run_ft_inference(f, dataset, feedback, train=False, run_id="1"):
    """
    Run inference and record the prediction result from Parameter Efficient FineTuning Adaptor
    f : PeftInferencer / ReftInferencer
    dataset : trainset / testset
    train : True / False
    run_info : "dft_pred_base" / "dft_pred_adaptor" / "reft_pred_base" / "reft_pred_adaptor"
    """
    if train:
        dset = dataset["train"]
    else:
        dset = dataset["test"]
    pb = tqdm(total=(len(dset)), desc = "Running reft adaptor inference")
    system_prompt = "Follow the instruction closely and provide your answer."
    pred_infos = []
    for data in dset:
        if isinstance(f, PeftInferencer):
            response = f.generate(prompt=data["prompt"])
        else:
            raise ValueError("Only PeftInferencer is supported for now")

        pred = response.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0]

        info = {"prompt": data["prompt"], "pred": pred, "gt": data["completion"]}
        pred_infos.append(info)
        pb.update(1)
    
    df = pd.DataFrame(pred_infos)
    try:
        repo_id = f.adaptor_id
    except:
        repo_id = f.reft_repo_id

    try:
        run_info = repo_id.split("/")[-1] + "_" + run_id
        df.to_csv(f"database/{feedback.file_name}/{run_info}.csv")
    except:
        print("Failed to save the result")
    return df


def run_peft_inference(f, dataset, feedback, train=False, run_id="1"):
    """
    Run inference and record the prediction result from Parameter Efficient FineTuning Adaptor
    f : PeftInferencer / ReftInferencer
    dataset : trainset / testset
    train : True / False
    run_info : "dft_pred_base" / "dft_pred_adaptor" / "reft_pred_base" / "reft_pred_adaptor"
    """
    if train:
        dset = dataset["train"]
    else:
        dset = dataset["test"]
        
    pred_infos = []    
    responses = f.batch_generate(prompts=dset["prompt"])
    for (data, response) in zip(dset, responses):
        response = data["response"]
        pred = response.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0]
        info = {"prompt": data["prompt"], "pred": pred, "gt": data["completion"]}
        pred_infos.append(info)
    
    df = pd.DataFrame(pred_infos)
    try:
        repo_id = f.adaptor_id
    except:
        repo_id = f.reft_repo_id

    try:
        run_info = repo_id.split("/")[-1] + "_" + run_id
        df.to_csv(f"database/{feedback.file_name}/{run_info}.csv")
    except:
        print("Failed to save the result")
    return df