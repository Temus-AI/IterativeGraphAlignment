import sys, os, json, glob, argparse, shutil
base_path =os.path.join(os.path.dirname(__file__), '..')
sys.path.append(base_path)

from src.dataset.feedback_utils_v2 import Feedback
from src.utils import ModelArguments, PeftArguments
from trl import SFTTrainer
from huggingface_hub import HfApi
from transformers import TrainingArguments, HfArgumentParser, AutoTokenizer
from src.custom_collator import (DataCollatorForCompletionOnlyLM_v2, 
                                 get_format_func, 
                                 get_teacher_format_func,
                                 infer_response_template)
import glob
import gc
import torch
from src.inference import PeftInferencer

def upload_folder_excluding_checkpoint(folder_path, repo_id):
    api = HfApi()

    # Remove Any Checkpoint Sub-Folders
    ckpt_folders = glob.glob(folder_path+"/checkpoint*")
    for ckpt_folder in ckpt_folders:
        shutil.rmtree(ckpt_folder)
    
    # Upload the folder, excluding the 'checkpoint' subfolder
    # Note: We use repo_type="model" for LoRA adaptors as well
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",  # LoRA adaptors use the "model" type
        allow_patterns=["*"],
    )

    print(f"Upload complete. Check your repository at: https://huggingface.co/{repo_id}")
    

def train_peft(arg_file, dataset, run_id: str = "1"):

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
    args.output_dir = f"{args.output_dir}_{run_id}"

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
            # dataset_text_field="text", # Question: I do NOT think 'text' is one of the key in the dataset ??
            formatting_func=formatting_prompt_func,
            data_collator=collator,
            packing=False,
            dataset_kwargs={
                # "add_special_tokens": False,  # We template with special tokens | Mistral v0.3 does not recognize this argument
                "append_concat_token": False, # No need to add additional separator token
            }
        )
    else:
        print("Algorithm not supported")

    trainer.train()


# Pending Dataset preparation (huggingface ver.)

# if __name__ == "__main__":
#     from huggingface_hub import login
#     HF_TOKEN = "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"
#     login(
#       token=HF_TOKEN,
#       add_to_git_credential=True
#     )
    
#     arg_file = "configs/config_sft_v1.json"
#     with open(arg_file, "r") as f:
#         arg_dict = json.load(f)
#     model_id = arg_dict["model_args"]["base_model_id"]
#     iter_n = 2
    
#     # Specific Feedback
#     feedback = Feedback(content = "You should not talk about Elephant")
    
#     # Prepare IGR dataset
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     dataset = feedback.prepare_IGR_dataset(tokenizer, ratio = [3, 1], iter_n=iter_n)

#     # Train the model
#     train_peft(arg_file, dataset, run_id="01")

#     # Clear CUDA cache
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
    
#     # Perform garbage collection
#     gc.collect()
    
    
#     # Unfortunately, full-precision LoRA Adaptor model is not doing Too Well now
#     adaptor_id = f"Ksgk-fy/IGR_iter{iter_n}_01"
#     f = PeftInferencer(adaptor_id, use_quant=False)
#     merge_model = f.model.merge_and_unload()
#     merge_model.push_to_hub(f"Ksgk-fy/IGR_iter{iter_n}_merge") # Merge and Push to Huggingface -- just so that we could serve with vLLM easily
#     f.tokenizer.push_to_hub(f"Ksgk-fy/IGR_iter{iter_n}_merge")