from src.IGRsample import IGRSampler
from src.config import IGRConfig
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='IGR Sampler')
parser.add_argument('--feedback_content', type=str, default="You should roleplay as a customer", help='Feedback content for the sampler')
parser.add_argument('--iteration', type=int, required=True, help='Current iteration number')
parser.add_argument("--id", type=str, default="", help="Run ID")
parser.add_argument("--helper_idx", type=int, default=0, help="Helper model index")
parser.add_argument("--load_slm", action="store_true", help="Load SLM")
parser.add_argument("--init_naive", action="store_true", help="Initialize naive responding")
parser.add_argument("--augment", action="store_true", help="Augment the prompt")
parser.add_argument("--do_star", action="store_true", help="Do STaR")
parser.add_argument("--check_alignment", action="store_true", help="Check alignment")
args = parser.parse_args()

# Use parsed arguments
feedback_content = args.feedback_content
config = IGRConfig(current_iteration=args.iteration, ID=args.id)

sampler = IGRSampler(feedback_content, config, roleplay=True, load_slm=args.load_slm, helper_idx=args.helper_idx, init_naive=args.init_naive, do_star=args.do_star)
if args.augment and not args.do_star:
    sampler.process_prompts(config.current_iteration, check_alignment=args.check_alignment)
elif not args.augment and not args.do_star:
    sampler.process_prompts(config.current_iteration, n_think=1, n_parrot=1, check_alignment=args.check_alignment)