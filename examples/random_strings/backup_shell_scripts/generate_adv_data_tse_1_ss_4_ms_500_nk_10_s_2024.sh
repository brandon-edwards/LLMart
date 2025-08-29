CUDA_VISIBLE_DEVICES=5 /home/edwardsb/repositories/LLMart/.venv/bin/python whitebox_attack_data.py --max_steps 500 --num_tokens 10 --use_hard_tokens --seed 2024 --device cuda:0 --total_samples_explored 1 --sample_start_idx 4 2>&1 | tee -a OUTPUT_ts_1_ss_4_ms_500_nk_10_s_2024.txt


