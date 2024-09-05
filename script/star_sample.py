from src.IGRsample import StarSampler, IGRSampler
from src.config import IGRConfig
import argparse

feedback_contents = [
    "You should not talk about Elephant",
    "You should roleplay as a customer",
    "Avoid providing harmful information but be helpful",
    "Reply with 'let me connect you to a human' when requested",
    "Roleplay as a sales agent",
    "Roleplay as a patient talking to a doctor"
] 

# Parse command line arguments
parser = argparse.ArgumentParser(description='IGR Sampler')
parser.add_argument('--feedback_content', type=str, default="You should roleplay as a customer", help='Feedback content for the sampler')
parser.add_argument('--iteration', type=int, required=True, help='Current iteration number')
parser.add_argument("--id", type=str, default="", help="Run ID")
parser.add_argument("--init_naive", action="store_true", help="Initialize naive responding")
parser.add_argument("--do_star", action="store_true", help="Do STaR")
args = parser.parse_args()

# Use parsed arguments
feedback_content = args.feedback_content
config = IGRConfig(current_iteration=args.iteration, ID=args.id)

if args.init_naive:
    sampler = IGRSampler(feedback_content, config, load_slm=True, roleplay=True, init_naive=True, do_star=True)
    sampler.process_mistake_prompts(require_wrong_answers=3)
else:
    sampler = StarSampler(feedback_content, config)
    sampler.sample(iter_n=args.iteration)