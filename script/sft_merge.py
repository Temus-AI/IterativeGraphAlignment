# Unfortunately vLLM / LMDeploy support for LoRA + Llama3 is bad for now ...
from src.config import IGRConfig
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='SFT Merger')
parser.add_argument("--id", type=str, default="", help="Run ID")
args = parser.parse_args()

config = IGRConfig(ID=args.id)

from src.inference import PeftInferencer
adaptor_id = config.sft_adaptor_id
f = PeftInferencer(adaptor_id, use_quant=False)
f.merge_and_upload(config.sft_model_id)