import os
import json
import time
from .graph import iterative_graph_prompting, FLAG_FOR_SUPERVISION, call_claude_api, self_evaluation, get_response, get_response_from_msg, call_calude_with_msg, get_oai_response
from tqdm import tqdm

def IGP_inference(query, feedback, model, roleplay=False, hint=""):
    """ 
    Inference on queries regarding current feedback using Iterative Graph Prompting
    """
    instruction = feedback.content
    sys_prompt = f"Instruction: {instruction}"
    if model == 'llama':
        initial_response = get_response(query, sys_prompt)
    elif model == 'claude':
        msg = [
            {"role": "user", "content": query},
        ]
        initial_response = call_calude_with_msg(msg, sys_prompt)
    elif model == 'gpt':
        initial_response = get_oai_response(query, sys_prompt)
        
    follow_instruction = self_evaluation(instruction, initial_response, model)
    
    if follow_instruction:
        print(f"Correct Answer Encountered for query: {query}")
        return initial_response
    
    print(f"Incorrect Answer Encountered for query: {query}")
    print("Initializing Heuristic Graph")
    
    response = iterative_graph_prompting(feedback, query, hint=hint, model=model, roleplay=roleplay)
            
    return response 
    


def inference_with_IGP(feedback, model="llama", roleplay=False):
    """ 
    Inference on queries regarding current feedback using Heuristic Graph
    - Self-evaluation at the beginning and the end 
    """
    
    response_file = f"database/{feedback.file_name}/IGP_responses_{model}.json"
    
    queries = feedback.prompts
    responses = []

    for query in tqdm(queries, desc="Running Iterative Graph Prompting"):
        response = IGP_inference(query, feedback, model, roleplay)        
        responses.append(response)        
        time.sleep(1)
        
    # Save responses to JSON file
    os.makedirs(os.path.dirname(response_file), exist_ok=True)
    with open(response_file, 'w') as f:
        json.dump(responses, f, indent=4)

    return [], responses


def cot_inference(query, instruction, model, roleplay=False):
        
    # Initial self-evaluation with CoT
    thought_prompt = f"Instruction: {instruction}\n\nPlease think step by step to answer the query. Show your step-by-step thinking."
    if model == 'llama':
        thought_response = get_response(query, thought_prompt)
    elif model == 'claude':
        msg = [
            {"role": "user", "content": query},
        ]
        thought_response = call_calude_with_msg(msg, thought_prompt)
    elif model == "gpt":
        thought_response = get_oai_response(query, thought_prompt)
    # Put the message history in context
        
    answer_prompt = f"Based on your previous reasoning, please state 'So the answer is:' followed by your final answer."
    message_history = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": thought_response},
        {"role": "user", "content": answer_prompt}
    ]   
    
    if model == 'llama':
        if roleplay:
            answer_response = get_response_from_msg(message_history, instruction)
        else:
            answer_response = get_response_from_msg(message_history, "You are a helpful assistant.")
    elif model == 'claude':
        if roleplay:
            answer_response = call_calude_with_msg(message_history, instruction)
        else:
            answer_response = call_calude_with_msg(message_history, "You are a helpful assistant.")
    elif model == "gpt":
        if roleplay:
            answer_response = get_oai_response(message_history, instruction)
        else:
            answer_response = get_oai_response(message_history, "You are a helpful assistant.")
    
    answer_response = answer_response.split("So the answer is:")[-1].strip()  # corner case: when there is only one element inside  
    
    return answer_response 


def inference_with_cot(feedback, model='llama', roleplay=False):
    """ 
    Inference on queries regarding current feedback using Chain of Thought (CoT)
    - Self-evaluation at the beginning and the end 
    - We adopt the method which Kojima took, which is to ask "So the answer is:" after the CoT
    """
    issue_file = f"database/{feedback.file_name}/cot_issue_prompts_{model}.json"
    response_file = f"database/{feedback.file_name}/cot_responses_{model}.json"
    if os.path.exists(issue_file) and os.path.exists(response_file):
        print(f"CoT results exists, skipping inference")
        return
    
    instruction = feedback.content
    queries = feedback.prompts
    issue_queries = []
    responses = []

    for query in tqdm(queries, desc="Running CoT Inference"):
        
        answer_response = ""

        for attempt in range(2):
            try:
                answer_response = cot_inference(query, instruction, model, roleplay)
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed. Error: {e}")
                if attempt == 0:
                    print("Waiting 30 seconds before retrying...")
                    time.sleep(30)
        
        # Evaluate the response
        follow_instruction = self_evaluation(feedback.content, answer_response, model)
        
        if follow_instruction:
            print(f"Correct Answer Encountered for query: {query}")
            responses.append(answer_response)
        else:
            print(f"Incorrect Answer Encountered for query: {query}")
            print(f"Answer: {answer_response}")
            issue_queries.append(query)
            responses.append(FLAG_FOR_SUPERVISION + " Issue response: " + answer_response)

    # Save issue queries to JSON file
    os.makedirs(os.path.dirname(issue_file), exist_ok=True)
    with open(issue_file, 'w') as f:
        json.dump(issue_queries, f, indent=4)
        
    # Save responses to JSON file
    os.makedirs(os.path.dirname(response_file), exist_ok=True)
    with open(response_file, 'w') as f:
        json.dump(responses, f, indent=4)

    return issue_queries, responses


def inference_naive(feedback, model="llama"):
    """ 
    Inference on queries naively 
    """
    issue_file = f"database/{feedback.file_name}/naive_issue_prompts_{model}.json"
    response_file = f"database/{feedback.file_name}/naive_responses_{model}.json"
    if os.path.exists(issue_file) and os.path.exists(response_file):
        print(f"Naive results exists, skipping inference")
        return
    
    instruction = feedback.content
    queries = feedback.prompts
    issue_queries = []
    responses = []

    for query in tqdm(queries, desc="Running Naive Inference"):
        sys_prompt = f"Instruction: {instruction}\n\nPlease answer the query directly."
        if model == 'llama':
            response = get_response(query, sys_prompt)
        elif model == 'claude':
            msg = [
                {"role": "user", "content": query}
            ]
            response = call_calude_with_msg(msg, sys_prompt)
        elif model == "gpt":
            response = get_oai_response(query, sys_prompt)
        
        # Evaluate the response
        follow_instruction = self_evaluation(feedback.content, response, model)
        
        if follow_instruction:
            print(f"Correct Answer Encountered for query: {query}")
        else:
            print("Error encountered")
            issue_queries.append(query)
                
        responses.append(response)

    # Save issue queries to JSON file
    os.makedirs(os.path.dirname(issue_file), exist_ok=True)
    with open(issue_file, 'w') as f:
        json.dump(issue_queries, f, indent=4)
        
    # Save responses to JSON file
    os.makedirs(os.path.dirname(response_file), exist_ok=True)
    with open(response_file, 'w') as f:
        json.dump(responses, f, indent=4)

    return issue_queries, responses   