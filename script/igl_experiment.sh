arg_file="configs/config_sft_v1.json"
feedback_content="You should roleplay as a customer"
run_id="hint"
python -m script.sample --feedback_content "$feedback_content" --iteration "$current_iteration" --id "$run_id" --helper_idx 0 --augment
python -m script.sample --feedback_content "$feedback_content" --iteration "$current_iteration" --id "$run_id" --helper_idx 1