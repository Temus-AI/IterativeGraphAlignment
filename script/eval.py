from src.config import IGRConfig
from src.IGRsample import IGRSampler
import argparse, os

# Parse command line arguments
parser = argparse.ArgumentParser(description='IGR Evaluator')
parser.add_argument('--feedback_content', type=str, default="You should not talk about Elephant", help='Feedback content for the sampler')
parser.add_argument('--iteration', type=int, required=True, help='Current iteration number')
parser.add_argument("--id", type=str, default="", help="Run ID")
parser.add_argument("--mode", type=str, default="star", help="star or sft or sail")
parser.add_argument("--check_alignment", action="store_true", help="Check alignment")
args = parser.parse_args()

# Use parsed arguments
feedback_content = args.feedback_content
config = IGRConfig(ID=args.id, current_iteration=args.iteration+1) # Evaluate on current iteration's model

# Sampler could be used directly to evaluate the model
model_name = ""
if args.mode == "star":
    model_name = config.BASE_STAR_MODEL
elif args.mode == "sft":
    model_name = config.BASE_SFT_MODEL
else:
    model_name = config.BASE_MODEL 
sampler = IGRSampler(feedback_content, config, roleplay=True, load_slm=True, init_naive=True, SLM_MODEL=model_name)
if args.mode == "star":
    sampler.eval_dir = f'database/{sampler.feedback.file_name}/Evaluate_{args.mode}_rationale_{sampler.id}_iter{sampler.current_iteration}.csv'
elif args.mode == "sft":
    sampler.eval_dir = f'database/{sampler.feedback.file_name}/Evaluate_{args.mode}_rationale_{sampler.id}.csv'
else:
    sampler.eval_dir = f"database/{sampler.feedback.file_name}/Evaluate_{args.mode}_rationale_{sampler.id}.csv"
sampler.eval_prompts(iter_n=args.iteration, check_alignment=args.check_alignment)