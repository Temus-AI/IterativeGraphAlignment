from src.config import IGRConfig
import subprocess

# Rest of your code remains the same
arg_file = "configs/config_sft_v1.json"
feedback_content = "You should roleplay as a customer"
run_id = "roleplay"

config = IGRConfig(ID=run_id)

# STaR experiment 
for current_iteration in range(1, config.MAX_ITER + 1):
    # STaR sampling at current iteration
    subprocess.run(['python', '-m', 'script.star_sample', '--feedback_content', feedback_content, '--iteration', str(current_iteration), "--id", run_id, "--init_naive"])
    subprocess.run(['python', '-m', 'script.star_sample', '--feedback_content', feedback_content, '--iteration', str(current_iteration), "--id", run_id])
    
     # IGR training at current iteration
    subprocess.run(['python', '-m', 'script.star_train', '--arg_file', arg_file, '--feedback_content', feedback_content, '--iteration', str(current_iteration), "--id", run_id])
    
    # Merge LoRA adaptor to avoid serving issue with vLLM: https://github.com/vllm-project/vllm/issues/6800
    subprocess.run(['python', '-m', 'script.star_merge', '--iteration', str(current_iteration), "--id", run_id])
    
    subprocess.run(['python', '-m', 'script.eval', '--feedback_content', feedback_content, '--iteration', str(current_iteration), '--id', run_id, "--mode", "star"])