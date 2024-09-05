from src.config import IGRConfig
import subprocess

# Rest of your code remains the same
arg_file = "configs/config_sft_v1.json"
feedback_content = "You should roleplay as a customer"
run_id = "roleplay"

subprocess.run(['python', '-m', 'script.sft_train', '--arg_file', arg_file, '--feedback_content', feedback_content, "--id", run_id])
subprocess.run(['python', '-m', 'script.sft_merge', "--id", run_id])
subprocess.run(['python', '-m', 'script.eval', '--feedback_content', feedback_content, '--id', run_id, "--mode", "sft", "--iteration", "1"])