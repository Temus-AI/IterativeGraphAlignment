from src.config import IGRConfig
import subprocess


# Rest of your code remains the same
arg_file = "configs/config_sft_v1.json"

# GSM8K experimentation
feedback_content = "Provide your thought and answer to the question from user. For example:\n        Thought: ...\n        Answer: ..."
check_alignment = False # For GSM8K, use quick alignment check to bypass self-confirmation bias
run_id = "gsm8k"

config = IGRConfig(ID=run_id)

# IGL experiment
for current_iteration in range(1, config.MAX_ITER + 1):
    # IGR sampling at current iteration
    subprocess.run(['python', '-m', 'script.sample', '--feedback_content', feedback_content, '--iteration', str(current_iteration), "--id", run_id, "--check_alignment", str(check_alignment), "--load_slm", "--init_naive"])
    subprocess.run(['python', '-m', 'script.sample', '--feedback_content', feedback_content, '--iteration', str(current_iteration), "--id", run_id, "--check_alignment", str(check_alignment), "--load_slm", "--augment"])
    subprocess.run(['python', '-m', 'script.sample', '--feedback_content', feedback_content, '--iteration', str(current_iteration), "--id", run_id, "--check_alignment", str(check_alignment), "--helper_idx", "0", "--augment"])
    subprocess.run(['python', '-m', 'script.sample', '--feedback_content', feedback_content, '--iteration', str(current_iteration), "--id", run_id, "--check_alignment", str(check_alignment), "--helper_idx", "1"])
    subprocess.run(['python', '-m', 'script.sample', '--feedback_content', feedback_content, '--iteration', str(current_iteration), "--id", run_id, "--check_alignment", str(check_alignment), "--helper_idx", "2"])
    subprocess.run(['python', '-m', 'script.sample', '--feedback_content', feedback_content, '--iteration', str(current_iteration), "--id", run_id, "--check_alignment", str(check_alignment), "--helper_idx", "3"])

    # IGR training at current iteration
    subprocess.run(['python', '-m', 'script.train', '--arg_file', arg_file, '--feedback_content', feedback_content, '--iteration', str(current_iteration), "--id", run_id])
    
    # Merge LoRA adaptor to avoid serving issue with vLLM: https://github.com/vllm-project/vllm/issues/6800
    subprocess.run(['python', '-m', 'script.merge', '--iteration', str(current_iteration), "--id", run_id])
    
    # Evaluation at current iteration 
    subprocess.run(['python', '-m', 'script.eval', '--feedback_content', feedback_content, '--iteration', str(current_iteration), '--id', run_id, "--mode", "igp"])
    print(f"------ IGR Iteration: {current_iteration} completed ------")