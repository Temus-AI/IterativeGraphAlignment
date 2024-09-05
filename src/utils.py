# Test with Maria conversation
# Test with the served Finetune checkpoints
from openai import OpenAI
import random
import numpy as np
import torch
from typing import Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DTYPES = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32
}
    
    
@dataclass
class ModelArguments:
    base_model_id: Optional[str] = "meta-llama/Meta-Llama-3-8B-Instruct"
    new_model_id: Optional[str] = "Yo-01"
    use_awq: Optional[bool] = False
    device_map: Optional[str] = "auto"
    attn_implementation: Optional[str] = "flash_attention_2"
    torch_dtype: Optional[torch.dtype] = torch.bfloat16

    def make(self, return_terminator: bool = False):
        """ 
        Make the args into LLM model
        """
        if self.use_awq: 
            # model = AutoAWQForCausalLM(
            #     self.base_model_id,
            #     fuse_layers=False
            # )
            raise NotImplementedError("AWQ LoRA is not supported")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                device_map=self.device_map,
                attn_implementation=self.attn_implementation,
                torch_dtype=self.torch_dtype
            )

        model_max_length = 4096
        
        if "Meta-Llama" in self.base_model_id:
            token_id = "meta-llama/Meta-Llama-3-8B-Instruct" # Special Gadget to patch bug with Llama3.1-8B model
        else:
            token_id = self.base_model_id
            
        tokenizer = AutoTokenizer.from_pretrained(
            token_id, 
            model_max_length=model_max_length, 
            padding_side="right", 
            use_fast=False)

        if "Meta-Llama-3-" in self.base_model_id:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        else:
            tokenizer.pad_token = tokenizer.eos_token
            terminators = tokenizer.eos_token_id
        
        if return_terminator:
            return model, tokenizer, terminators
        else:
            return model, tokenizer
    
        
@dataclass
class PeftArguments:
    lora_alpha: Optional[int] = 128
    lora_dropout: Optional[float] = 0.05
    r: Optional[int] = 256
    bias: Optional[str] = "none"
    target_modules: Optional[str] = "all"
    task_type: Optional[str] = "CAUSAL_LM"
    # save_embedding_layers: Optional[bool] = True

    def make(self):
        return LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.r,
            bias=self.bias,
            target_modules=self.target_modules,
            task_type=self.task_type
            # save_embedding_layers=self.save_embedding_layers
        )


arg_to_repo = lambda arg_file: arg_file.split("/config_")[-1].replace(".json", "_elvf")
# repo_to_arg = lambda repo_id: f"configs/config_{repo_id.split("Ksgk-fy/")[-1].replace('_elvf', '')}.json"
def repo_to_arg(repo_id):
    repo_name = repo_id.split("Ksgk-fy/")[-1]
    repo_name_to_arg = lambda repo_id: f"configs/config_{repo_id.replace('_elvf', '')}.json"
    return repo_name_to_arg(repo_name)


def formattting_query_prompt_func_with_sys(prompt, sys_prompt,
                                           tokenizer,
                                           completion = "####Dummy-Answer"):
    """ 
    Formatting response according to specific llama3 chat template
    """
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion}
    ]
    format_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    query_prompt = format_prompt.split(completion)[0]
    return query_prompt

def formatting_query_prompt(message_history, 
                             sys_prompt,
                             tokenizer,
                             completion = "####Dummy-Answer"):
    """ 
    Formatting response according to specific llama3 chat template
    """
    messages = [{"role":"system", "content":sys_prompt}]
    messages.extend(message_history)
    messages.extend([{"role": "assistant", "content": completion}])
    format_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    query_prompt = format_prompt.split(completion)[0]
    return query_prompt

def get_response_from_finetune_checkpoint(format_prompt, do_print=True):
    """
    - Using vLLM to serve the fine-tuned Llama3 checkpoint
    - v6 full precision checkpoint is adopted here
    """
    # Serving bit of the client
    client = OpenAI(api_key="EMPTY", base_url="http://43.218.77.178:8000/v1")    
    # model_name = "Ksgk-fy/ecoach_philippine_v6_product_merge"
    # model_name = "Ksgk-fy/genius_merge_v1"
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # Streaming bit of the client
    stream = client.completions.create(
                model=model_name,
                prompt=format_prompt,
                max_tokens=512,
                temperature=0.0,
                stop=["<|eot_id|>"],
                stream=True,
                extra_body={
                    "repetition_penalty": 1.1,
                    "length_penalty": 1.0,
                    "min_tokens": 0,
                },
            )

    if do_print:
        1 + 1
        # print("Maria: ", end="")
    response_text = ""
    for response in stream:
        txt = response.choices[0].text
        if txt == "\n":
            continue
        if do_print:
            print(txt, end="")
        response_text += txt
    response_text += "\n"
    if do_print:
        print("")
    return response_text


def get_response_from_base(format_prompt, do_print=True, temperature=0.0, max_tokens=512, prefix=""):
    """ 
    - Using vLLM 
    """
    client = OpenAI(api_key="EMPTY", base_url="http://ec2-13-229-24-160.ap-southeast-1.compute.amazonaws.com:8000/v1")
    model_name = "MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ"
    # Streaming bit of the client
    stream = client.completions.create(
                model=model_name,
                prompt=format_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|eot_id|>"],
                stream=True,
                extra_body={
                    "repetition_penalty": 1.1,
                    "length_penalty": 1.0,
                    "min_tokens": 0,
                },
            )

    if do_print:
        print("Llama3: ", end="")
    response_text = prefix if prefix else ""
    for response in stream:
        txt = response.choices[0].text
        if txt == "\n":
            continue
        if do_print:
            print(txt, end="")
        response_text += txt
    response_text += "\n"
    if do_print:
        print()  # Add a newline after the complete response
    return response_text


from transformers import AutoTokenizer
from os import getenv
from huggingface_hub import login
login(getenv("HF_TOKEN"))
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=False)


# SmartAss OnBoard
def parse_lex_transcript(soup):
    transcript_segments = soup.find_all('div', class_='ts-segment')
    transcript = ""
    curr_speaker = ""
    curr_response = ""

    for segment in transcript_segments:
        speaker = segment.find('span', class_='ts-name').get_text(strip=True)
        timestamp = segment.find('span', class_='ts-timestamp').get_text(strip=True)
        text = segment.find('span', class_='ts-text').get_text(strip=True)

        if curr_speaker == "":
            curr_speaker = speaker
            curr_response = text 
        elif curr_speaker != speaker:
            transcript += f"\n\n{curr_speaker}: {curr_response}"
            curr_speaker = speaker
            curr_response = text
        elif curr_speaker == speaker:
            curr_response = curr_response.strip() + f" {text}"  

    # Save the trnascript into txt files 
    with open("smart-ass/sm1.txt", "w") as file:
        file.write(transcript.strip())

# Random Rephraser
def get_rand_seq(tokenizer, min_len=1, max_len=20):
    rand_len = random.randint(min_len, max_len)
    rand_seq_ids = np.random.choice(tokenizer.vocab_size, rand_len)
    rand_seq = tokenizer.decode(rand_seq_ids)
    return rand_seq


def get_rand_sequence(tokenizer, min_len=1, max_len=20, rephrase=True):
    rand_len = random.randint(min_len, max_len)
    rand_seq_ids = np.random.choice(tokenizer.vocab_size, rand_len)
    rand_seq = tokenizer.decode(rand_seq_ids)
    if rephrase:
        rand_seq = parse_rephrase_utterance(rand_seq)
    return rand_seq


def process_prompt_chain(prompt_chain, sys_prompt, noise=False):
    message_history = []
    if noise:
        random_prompt = get_rand_sequence(tokenizer, min_len=1, max_len=3, rephrase=False)
        full_prompt = random_prompt + " " + sys_prompt
    else:
        random_prompt = ""
        full_prompt = sys_prompt
    for prompt in prompt_chain:
        message_history.append({"role": "user", "content": prompt})
        format_prompt = formatting_query_prompt(message_history, full_prompt, tokenizer)
        response = get_response_from_finetune_checkpoint(format_prompt=format_prompt, do_print=False)
        message_history.append({"role": "assistant", "content": response})

    return {"response": response, "random_prompt": random_prompt}


def generate_rephrase_prompt(original_text):
    rephrase_prompt = f"""
    Please rephrase the following text while maintaining its original meaning:

    Original text: "{original_text}"

    Rephrased version:
    """
    return rephrase_prompt

def parse_rephrase_utterance(original_text):
    # Generate the rephrase prompt
    prompt = generate_rephrase_prompt(original_text)
    
    message_history = []
    # Append the prompt to the message history
    
    message_history.append({"role": "user", "content": prompt})

    # Format the prompt using the provided function
    sys_prompt = ""
    format_prompt = formatting_query_prompt(message_history, sys_prompt, tokenizer)
    
    # Get the response from the fine-tuned checkpoint
    response = get_response_from_finetune_checkpoint(format_prompt=format_prompt)
    
    # Append the response to the message history
    message_history.append({"role": "assistant", "content": response})
    
    # Return the rephrased text
    return response

# Evaluation Bits
def evaluate_response(response, sys_prompt):
    
    evaluation_prompt = f"""
    Evaluate the following response for:
    1. Factuality (Is the information correct?)
    2. Adherence to the instruction: "{sys_prompt}"

    Response: "{response}"

    Provide your evaluation in the following format:
    Factual: [Yes/No]
    Follows Instruction: [Yes/No]
    Explanation: [Your explanation]
    """
    t = 0
    complete_evaluation = False
    while t < 4 and (not complete_evaluation):
        try:
            message_history = []
            message_history.append({"role": "user", "content": evaluation_prompt})
            format_prompt = formatting_query_prompt(message_history, sys_prompt, tokenizer)
            evaluation = get_response_from_finetune_checkpoint(format_prompt=format_prompt, do_print=False)
            message_history.append({"role": "assistant", "content": evaluation})
            # Parse the evaluation result
            lines = evaluation.split('\n')
            factual = lines[0].split(': ')[1].strip().lower() == 'yes'
            follows_instruction = lines[1].split(': ')[1].strip().lower() == 'yes'
            explanation = lines[2].split(': ')[1].strip()
            complete_evaluation = True
        except:
            t += 1
            factual = False
            follows_instruction = False
            explanation = "No evaluation provided"
    return factual, follows_instruction, explanation

def generate_self_extrapolation_prompt(issue):
    if issue == "factuality":
        return "How can I ensure my response is more factually accurate?"
    elif issue == "instruction":
        return "How can I better adhere to the given instruction in my response?"
    else:
        return "How can I improve my response?"
    
def generate_followup_question(response):
    sys_prompt = "You are a curious assistant. Generate a follow-up question to clarify or expand on the given response."
    prompt = f"Based on this response: '{response}', what follow-up question should we ask to get more information or clarification?"
    
    message_history = [{"role": "user", "content": prompt}]
    format_prompt = formatting_query_prompt(message_history, sys_prompt, tokenizer)
    followup_question = get_response_from_base(format_prompt, do_print=False)
    return followup_question.strip()


def get_response(prompt, sys_prompt):
    message_history = [{"role": "user", "content": prompt}]
    format_prompt = formatting_query_prompt(message_history, sys_prompt, tokenizer)
    response = get_response_from_base(format_prompt)
    return response 

def get_response_from_msg(message_history, sys_prompt):
    """ 
    Message History as list of dict
    """
    format_prompt = formatting_query_prompt(message_history, sys_prompt, tokenizer)
    response = get_response_from_base(format_prompt)
    return response