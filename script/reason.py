import os
import json
from src.dataset.feedback_utils_v2 import Feedback
from src.dataset.format_v2 import to_distill_sft
from src.graph import visualize_graph, call_claude_api
from src.hg import inference_with_cot, inference_naive, inference_with_IGP

def do_stat(evaluations):
    total_evaluations = len(evaluations)

    true_count = evaluations.count(True)
    false_count = evaluations.count(False)

    true_percentage = (true_count / total_evaluations) * 100
    false_percentage = (false_count / total_evaluations) * 100

    print(f"Total evaluations: {total_evaluations}")
    print(f"True count: {true_count} ({true_percentage:.2f}%)")
    print(f"False count: {false_count} ({false_percentage:.2f}%)")
    
    
def analyze_performance(feedback, model):
    
    if model == "llama":
        modes = ["naive", "cot"]
    else:
        # modes = ["naive", "cot", "IGP"]
        modes = ["IGP"]
    
    for mode in modes:
        print("Mode: ", mode)
        evaluations, explanations = feedback.evaluate_alignment(mode, model, call_claude_api)
        
        
        file_path = f"database/{feedback.file_name}/{mode}_evaluations_{model}.json"
        reports = []
        for prompt, evaluation, explanation in zip(feedback.prompts, evaluations, explanations):
            response = feedback.load_response(prompt, mode, model)
            reports.append({"query": prompt, "response": response, "evaluation": evaluation, "explanation": explanation})
        with open(file_path, "w") as f:
            f.write(json.dumps(reports))
            
        do_stat(evaluations)
        
        
# def analyze_reason_performance(feedback: Feedback, mode: str, model: str = "llama"):
#     """ 
#     Mode: Naive | Cot | GraphReasoner
#     """
    
#     file_path = f"database/{feedback.file_name}/{mode}_issue_prompts_{model}.json"
#     if os.path.exists(file_path):
#         with open(file_path, 'r') as f:
#             issue_prompts = json.load(f)
#     else:
#         issue_prompts = []
    
#     # Print the percentage of correct cases & error rate 
#     total_prompts = len(feedback.prompts)
#     correct_cases = total_prompts - len(issue_prompts)
#     correct_percentage = (correct_cases / total_prompts) * 100

#     # Calculate the error rate
#     error_rate = (len(issue_prompts) / total_prompts) * 100

#     print(f"For {mode} mode:")
#     print(f"Percentage of correct cases: {correct_percentage:.2f}%")
#     print(f"Error rate: {error_rate:.2f}%")

#     # Human Annotation Collection
#     actually_correct = []

#     for idx, query in enumerate(issue_prompts):
#         img, graph = feedback.load_graph_and_image(query)
#         response = feedback.load_response(query, mode, model)
        
#         print(f"\n----- Query {idx} -----")
#         print("------ Logical Graph ------")
#         visualize_graph(graph)
#         print("----- Response ------")
#         print(response)
#         print("\n----- Annotation -----")
#         user_input = input("Is this response okay? (y/n): ").lower()
        
#         if user_input == 'y':
#             actually_correct.append(idx)
        
#         # Clear the output
#         os.system('cls' if os.name == 'nt' else 'clear')

#     print("Annotation complete. Indices of correct responses:", actually_correct)
#     really_wrong = [idx for idx in range(len(issue_prompts)) if idx not in actually_correct]
#     print("Accuracy Rate: ", len(really_wrong)/len(feedback.prompts)*100, "%")

#     # Update the issue_prompts json file
#     updated_issue_prompts = [issue_prompts[idx] for idx in really_wrong]
#     with open(file_path, 'w') as f:
#         json.dump(updated_issue_prompts, f, indent=4)
    
#     print(f"Updated {file_path} with {len(updated_issue_prompts)} remaining issue prompts.")
        

feedback_contents = [
    "You should not talk about Elephant",
    "You should roleplay as a customer",
    "Avoid providing harmful information but be helpful",
    "Reply with 'let me connect you to a human' when requested",
    "Roleplay as a sales agent",
    "Roleplay as a patient talking to a doctor"
] 

import argparse

# Argument parser to get the feedback_idx
parser = argparse.ArgumentParser(description='Process feedback index.')
parser.add_argument('--idx', type=int, default=1, help='Index of the feedback content to use')
args = parser.parse_args()

feedback_idx = args.idx
feedback = Feedback(content = feedback_contents[feedback_idx])
roleplay = (feedback_idx in [1,4,5])

print(" ---- RolePlay Scenario: ", roleplay)

# Llama is run with Naive & CoT just to understand the performance before IGL procedure 
# for test_model in ["claude"]: # For iterative graph prompting, we focus on Sonnet 3.5 as it reacts the strongest to IGP approach
for test_model in ["gpt"]:
    print("Running Inference with model: ", test_model)
    # Comparison of three inference techniques | using same model (VLM) -- Claude Sonnet in this case
    # inference_naive(feedback, model=test_model)
    # inference_with_cot(feedback, model=test_model, roleplay=roleplay)
    if test_model == "llama": # test-based LLM should not use IGP here ...
        continue
    inference_with_IGP(feedback, model=test_model, roleplay=roleplay)

# Human annotation required to go through the issues cases & validate them | Require Mannual Attention here
for test_model in ["gpt"]:
    analyze_performance(feedback, test_model)