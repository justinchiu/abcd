#!/bin/bash
##SBATCH -J ws-semisup                       # Job name
##SBATCH -J oracle-sent-dummy                       # Job name
#SBATCH -J sent256                       # Job name
##SBATCH -J sent512                       # Job name
#SBATCH -o slurm/sent_%j.out          # output file (%j expands to jobID)
#SBATCH -e slurm/sent_%j.err          # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email
#SBATCH --mail-user=jtc257@cornell.edu       # Email address to send results to.
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 2                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=16000                          # server memory requested (per node)
#SBATCH -t 48:00:00                           # Time limit (hh:mm:ss)
##SBATCH --nodelist=rush-compute-03 # Request partition
##SBATCH --nodelist=rush-compute-02 # Request partition
##SBATCH --nodelist=rush-compute-01 # Request partition
#SBATCH --partition=rush,gpu # Request partition
##SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --gres=gpu:a6000:1                  # Type/number of GPUs needed

. "/home/jtc257/anaconda3/etc/profile.d/conda.sh"
source /home/jtc257/.bashrc
source /home/jtc257/scripts/env.sh
py113env

#python run_ws_answer_model.py --num_z_samples 16 --batch_size 2 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 125 --gradient_accumulation_steps 8 --subsample subflow --subsample_k 15 --subsample_steps 250 --subsample_passes 4 --subsampled_batch_size 16
#python run_ws_answer_model.py --num_z_samples 16 --batch_size 2 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 125 --gradient_accumulation_steps 8 --subsample subflow --subsample_k 20 --subsample_steps 250 --subsample_passes 4 --subsampled_batch_size 16
#python run_ws_answer_model.py --num_z_samples 16 --batch_size 2 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 125 --gradient_accumulation_steps 8 --subsample subflow --subsample_k 15 --subsample_steps 250 --subsample_passes 4 --subsampled_batch_size 16

# conditional
#python run_ws_answer_model.py --num_z_samples 16 --batch_size 2 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 124 --gradient_accumulation_steps 8 --subsample subflow --subsample_k 10 --subsample_steps 250 --subsample_passes 4 --subsampled_batch_size 2 --subsample_gradient_accumulation_steps 8 --subsample_obj conditional
#python run_ws_answer_model.py --num_z_samples 16 --batch_size 2 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 124 --gradient_accumulation_steps 8 --subsample subflow --subsample_k 15 --subsample_steps 250 --subsample_passes 4 --subsampled_batch_size 2 --subsample_gradient_accumulation_steps 8 --subsample_obj conditional
#python run_ws_answer_model.py --num_z_samples 16 --batch_size 2 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 124 --gradient_accumulation_steps 8 --subsample subflow --subsample_k 20 --subsample_steps 250 --subsample_passes 4 --subsampled_batch_size 2 --subsample_gradient_accumulation_steps 8 --subsample_obj conditional

# large sample conditional
#python run_ws_answer_model.py --num_z_samples 16 --batch_size 2 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 124 --gradient_accumulation_steps 8 --subsample subflow --subsample_k 200 --subsample_steps 250 --subsample_passes 2 --subsampled_batch_size 2 --subsample_gradient_accumulation_steps 8 --subsample_obj conditional
# large sample joint
#python run_ws_answer_model.py --num_z_samples 16 --batch_size 2 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 124 --gradient_accumulation_steps 8 --subsample subflow --subsample_k 200 --subsample_steps 250 --subsample_passes 2 --subsampled_batch_size 16 --subsample_gradient_accumulation_steps 1 --subsample_obj joint

# oracle sent
# no init from previous
#python run_oracle_sent_model.py --batch_size 8 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 119 --gradient_accumulation_steps 2
#python run_oracle_sent_model.py --batch_size 2 --eval_batch_size 4 --max_length 512 --eval_steps 250 --epoch 10 --prefix 119 --gradient_accumulation_steps 8

# init from previous
#python run_oracle_sent_model.py --batch_size 8 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 119 --gradient_accumulation_steps 2 --init_from_previous
#python run_oracle_sent_model.py --batch_size 2 --eval_batch_size 4 --max_length 512 --eval_steps 250 --epoch 10 --prefix 119 --gradient_accumulation_steps 8 --init_from_previous

# oracle sent with info tradeoff
#python run_oracle_sent_info_model.py --batch_size 2 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 21 --gradient_accumulation_steps 8 --max_turns 24 --max_turn_length 16
# with init-from-previous
#python run_oracle_sent_info_model.py --batch_size 2 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 21 --gradient_accumulation_steps 8 --max_turns 24 --max_turn_length 16 --init_from_previous
# longer turns
#python run_oracle_sent_info_model.py --batch_size 1 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 21 --gradient_accumulation_steps 16 --max_turns 32 --max_turn_length 32

# oracle sent, init from previous, dummy align
#python run_oracle_sent_model.py --batch_size 8 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 26 --gradient_accumulation_steps 2 --init_from_previous --dummy_step
#python run_oracle_sent_model.py --batch_size 2 --eval_batch_size 4 --max_length 512 --eval_steps 250 --epoch 10 --prefix 26 --gradient_accumulation_steps 8 --init_from_previous --dummy_step
# oracle sent dummy align
#python run_oracle_sent_model.py --batch_size 4 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 26 --gradient_accumulation_steps 4 --dummy_step
#python run_oracle_sent_model.py --batch_size 2 --eval_batch_size 4 --max_length 512 --eval_steps 250 --epoch 10 --prefix 26 --gradient_accumulation_steps 8 --dummy_step
# oracle sent init from previous rerun
#python run_oracle_sent_model.py --batch_size 2 --eval_batch_size 4 --max_length 512 --eval_steps 250 --epoch 10 --prefix 26 --gradient_accumulation_steps 8 --init_from_previous

# oracle sent bart-large ml256 2/13
#python run_oracle_sent_model.py --batch_size 2 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 213 --gradient_accumulation_steps 8 --monotonic_train --max_turns 128 --answer_model_dir facebook/bart-large
#python run_oracle_sent_model.py --batch_size 2 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 213 --gradient_accumulation_steps 8 --monotonic_train --max_turns 128 --answer_model_dir facebook/bart-large --learning_rate 1e-5

# oracle sent bart-base ml512 2/20
#python run_oracle_sent_model.py --batch_size 1 --eval_batch_size 4 --max_length 512 --eval_steps 250 --epoch 10 --prefix 213 --gradient_accumulation_steps 16 --max_turns 128 --decoder_turn_attention --learning_rate 2e-5
#python run_oracle_sent_model.py --batch_size 1 --eval_batch_size 4 --max_length 512 --eval_steps 250 --epoch 10 --prefix 213 --gradient_accumulation_steps 16 --max_turns 128 --decoder_turn_attention --learning_rate 1e-5

# oracle sent bart-base ml256 2/20
#python run_oracle_turn_model.py --batch_size 16 --eval_batch_size 16 --max_length 32 --max_step_length 128 --eval_steps 250 --epoch 10 --prefix 220 --gradient_accumulation_steps 8 --max_turns 128 --answer_model_dir facebook/bart-large --learning_rate 1e-5

# doc step model ml256 2/28
#python run_doc_step_model.py --batch_size 1 --eval_batch_size 4 --max_length 256 --max_step_length 128 --eval_steps 250 --epoch 10 --prefix 228 --gradient_accumulation_steps 16 --max_turns 128 --learning_rate 1e-5 --num_z_samples 4 --monotonic_train --decoder_turn_attention --init_from_previous

# longer doc model 3/1
#python run_ws_answer_model.py --num_z_samples 16 --batch_size 1 --eval_batch_size 2 --max_length 384 --eval_steps 250 --epoch 10 --prefix 31 --gradient_accumulation_steps 16
#python run_ws_answer_model.py --num_z_samples 12 --batch_size 1 --eval_batch_size 2 --max_length 448 --eval_steps 250 --epoch 10 --prefix 31 --gradient_accumulation_steps 16
#python run_ws_answer_model.py --num_z_samples 4 --batch_size 1 --eval_batch_size 2 --max_length 512 --eval_steps 250 --epoch 10 --prefix 31 --gradient_accumulation_steps 16

# doc step model ml256 3/1
#python run_doc_step_model.py --batch_size 1 --eval_batch_size 4 --max_length 256 --max_step_length 128 --eval_steps 250 --epoch 10 --prefix 31 --gradient_accumulation_steps 16 --max_turns 128 --learning_rate 1e-5 --num_z_samples 2 --monotonic_train --decoder_turn_attention --init_from_previous


# oracle sent bart-base ml512 3/5
#python run_oracle_sent_model.py --batch_size 1 --eval_batch_size 4 --max_length 512 --eval_steps 250 --epoch 10 --prefix 35 --gradient_accumulation_steps 16 --max_turns 128 --learning_rate 2e-5
python run_oracle_sent_model.py --batch_size 1 --eval_batch_size 4 --max_length 512 --eval_steps 250 --epoch 10 --prefix 35 --gradient_accumulation_steps 16 --max_turns 128 --learning_rate 1e-5
