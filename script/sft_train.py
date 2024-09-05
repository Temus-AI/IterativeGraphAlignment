from src.IGRtrain import IGRTrainer
from src.config import IGRConfig
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='IGR Sampler')
parser.add_argument('--feedback_content', type=str, default="You should roleplay as a customer", help='Feedback content for the sampler')
parser.add_argument('--arg_file', type=str, default="configs/config_sft_v1.json", help='Path to the argument file')
parser.add_argument("--id", type=str, default="", help="Run ID")
args = parser.parse_args()

# Use parsed arguments
feedback_content = args.feedback_content
config = IGRConfig(ID=args.id)

# Train & Save | debugging | Convert to subprocess.run() to avoid GPU memory release issue
trainer = IGRTrainer(config)
trainer.run_sft(args.arg_file, feedback_content)