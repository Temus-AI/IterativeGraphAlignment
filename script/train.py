from src.IGRtrain import IGRTrainer
from src.config import IGRConfig
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='IGR Sampler')
parser.add_argument('--feedback_content', type=str, default="You should roleplay as a customer", help='Feedback content for the sampler')
parser.add_argument('--iteration', type=int, required=True, help='Current iteration number')
parser.add_argument('--arg_file', type=str, default="configs/config_sft_v1.json", help='Path to the argument file')
parser.add_argument("--id", type=str, default="", help="Run ID")
args = parser.parse_args()

# Use parsed arguments
feedback_content = args.feedback_content
config = IGRConfig(ID=args.id, current_iteration=args.iteration)

# Train & Save | debugging | Convert to subprocess.run() to avoid GPU memory release issue
trainer = IGRTrainer(config)
trainer.run(args.arg_file, feedback_content)