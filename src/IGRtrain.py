import sys, os, json, glob, argparse, shutil
base_path =os.path.join(os.path.dirname(__file__), '..')
sys.path.append(base_path)

from .dataset.feedback_utils_v2 import Feedback
from .utils import ModelArguments, PeftArguments
from trl import SFTTrainer
from huggingface_hub import HfApi, login
from transformers import TrainingArguments, HfArgumentParser, AutoTokenizer
from src.custom_collator import (DataCollatorForCompletionOnlyLM_v2, 
                                 get_format_func, 
                                 get_teacher_format_func,
                                 infer_response_template)
import glob
import gc
import torch
from src.inference import PeftInferencer
import json
import shutil
import os
from .config import * 


class IGRTrainer:
    
    def __init__(self, config: IGRConfig):
        self.config = config
        login(token=self.config.HF_TOKEN, add_to_git_credential=True)

    def upload_folder_excluding_checkpoint(self, folder_path, repo_id):
        api = HfApi()

        # Remove Any Checkpoint Sub-Folders
        ckpt_folders = glob.glob(folder_path+"/checkpoint*")
        for ckpt_folder in ckpt_folders:
            shutil.rmtree(ckpt_folder)
        
        # Upload the folder, excluding the 'checkpoint' subfolder
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=["*"],
        )

        print(f"Upload complete. Check your repository at: https://huggingface.co/{repo_id}")
        
    def train_peft(self, arg_file, dataset, output_dir):
        # Load Argument Configuration & Get the Modes etc.
        with open(arg_file, "r") as f:
            arg_dict = json.load(f)

        # Load Model
        model_arg_parser = HfArgumentParser((ModelArguments,))
        model_args: ModelArguments = model_arg_parser.parse_dict(arg_dict["model_args"])[0]
        model, tokenizer = model_args.make()

        # Load LoRA arguments
        peft_args: PeftArguments = HfArgumentParser((PeftArguments,)).parse_dict(arg_dict["lora_args"])[0]
        peft_config = peft_args.make()

        # Load Training Arguments
        args = HfArgumentParser((TrainingArguments,)).parse_dict(arg_dict["training_args"])[0]
        args.output_dir = output_dir

        # Trainer Preparation
        response_template = infer_response_template(tokenizer)
        collator = DataCollatorForCompletionOnlyLM_v2(response_template, tokenizer=tokenizer)
        formatting_prompt_func = get_format_func(tokenizer)
        teacher_formatting_prompt_func = get_teacher_format_func(tokenizer)

        algo = arg_dict["algorithm"]
        max_seq_length = 1024

        if algo == "sft":
            args.remove_unused_columns=True
            trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=dataset["train"],
                peft_config=peft_config,
                max_seq_length=max_seq_length,
                tokenizer=tokenizer,
                formatting_func=formatting_prompt_func,
                data_collator=collator,
                packing=False,
                dataset_kwargs={
                    "append_concat_token": False,
                }
            )
        else:
            print("Algorithm not supported")

        trainer.train()
        
    def prep_dataset(self, arg_file, feedback_content): # Quick question, how does run_id affect anything here? it does not and this is wrong (!)
        with open(arg_file, "r") as f:
            arg_dict = json.load(f)
        model_id = arg_dict["model_args"]["base_model_id"]
        
        # Specific Feedback
        feedback = Feedback(content=feedback_content)
        
        # Prepare IGR dataset
        if "Meta-Llama" in model_id: # We are almost allergic to this word ...
            token_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        else:
            token_id = model_id 
                       
        tokenizer = AutoTokenizer.from_pretrained(token_id)
        dataset = feedback.prepare_IGR_dataset(tokenizer, iter_n=self.config.current_iteration, ID=self.config.ID) # KeyChange here
        return dataset
    
    def prep_sft_dataset(self, arg_file, feedback_content):
        with open(arg_file, "r") as f:
            arg_dict = json.load(f)
        model_id = arg_dict["model_args"]["base_model_id"]
        
        # Specific Feedback
        feedback = Feedback(content=feedback_content)
        
        # Prepare IGR dataset
        if "Meta-Llama" in model_id: # We are almost allergic to this word ...
            token_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        else:
            token_id = model_id 
                       
        tokenizer = AutoTokenizer.from_pretrained(token_id)
        dataset = feedback.prepare_SFT_dataset(tokenizer)
        return dataset
    
    def prep_star_dataset(self, arg_file, feedback_content):
        with open(arg_file, "r") as f:
            arg_dict = json.load(f)
        model_id = arg_dict["model_args"]["base_model_id"]
        
        # Specific Feedback
        feedback = Feedback(content=feedback_content)
        
        # Prepare IGR dataset
        if "Meta-Llama" in model_id: # We are almost allergic to this word ...
            token_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        else:
            token_id = model_id 
                       
        tokenizer = AutoTokenizer.from_pretrained(token_id)
        dataset = feedback.prepare_STaR_dataset(tokenizer, iter_n=self.config.current_iteration, ID=self.config.ID)
        return dataset
        
        

    def run(self, arg_file, feedback_content):
        
        # Prepare IGR training dataset
        dataset = self.prep_dataset(arg_file, feedback_content)
        print("------ IGR Dataset Preparation Complete ------")

        # Train the model
        adaptor_output_dir = self.config.adaptor_output_dir
        self.train_peft(arg_file, dataset, adaptor_output_dir)
        
    def run_sft(self, arg_file, feedback_content):
        # Prepare IGR training dataset
        dataset = self.prep_sft_dataset(arg_file, feedback_content)
        print("------ IGR Dataset Preparation Complete ------")
        
        # Train the model
        adaptor_output_dir = self.config.sft_adaptor_output_dir
        self.train_peft(arg_file, dataset, adaptor_output_dir)
        
    def run_star(self, arg_file, feedback_content):
        # Prepare IGR training dataset
        dataset = self.prep_star_dataset(arg_file, feedback_content)
        print("------ IGR Dataset Preparation Complete ------")
        
        # Train the model
        adaptor_output_dir = self.config.star_adaptor_output_dir
        self.train_peft(arg_file, dataset, adaptor_output_dir)