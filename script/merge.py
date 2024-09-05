# Unfortunately vLLM / LMDeploy support for LoRA + Llama3 is bad for now ...
from src.config import IGRConfig
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='IGR Merger')
parser.add_argument('--iteration', type=int, required=True, help='Current iteration number')
parser.add_argument("--id", type=str, default="", help="Run ID")
args = parser.parse_args()

config = IGRConfig(ID=args.id, current_iteration=args.iteration)

from src.inference import PeftInferencer
adaptor_id = config.adaptor_id
f = PeftInferencer(adaptor_id, use_quant=False)
f.merge_and_upload(config.model_id)