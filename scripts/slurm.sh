#!/bin/bash
#SBATCH -J sent256                       # Job name
##SBATCH -J sent512                       # Job name
#SBATCH -o slurm/sent_%j.out          # output file (%j expands to jobID)
#SBATCH -e slurm/sent_%j.err          # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email
#SBATCH --mail-user=jtc257@cornell.edu       # Email address to send results to.
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 2                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=32000                          # server memory requested (per node)
#SBATCH -t 24:00:00                           # Time limit (hh:mm:ss)
##SBATCH --nodelist=rush-compute-02 # Request partition
##SBATCH --nodelist=rush-compute-01 # Request partition
#SBATCH --partition=rush,gpu # Request partition
##SBATCH --gres=gpu:1                  # Type/number of GPUs needed
##SBATCH --partition=gpu,rush # Request partition
#SBATCH --gres=gpu:a6000:1                  # Type/number of GPUs needed

. "/home/jtc257/anaconda3/etc/profile.d/conda.sh"
source /home/jtc257/.bashrc
source /home/jtc257/scripts/env.sh
py113env

# no init from previous
python run_oracle_sent_model.py --batch_size 8 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 119 --gradient_accumulation_steps 2
#python run_oracle_sent_model.py --batch_size 2 --eval_batch_size 4 --max_length 512 --eval_steps 250 --epoch 10 --prefix 119 --gradient_accumulation_steps 8

# init from previous
#python run_oracle_sent_model.py --batch_size 8 --eval_batch_size 4 --max_length 256 --eval_steps 250 --epoch 10 --prefix 119 --gradient_accumulation_steps 2 --init_from_previous
#python run_oracle_sent_model.py --batch_size 2 --eval_batch_size 4 --max_length 512 --eval_steps 250 --epoch 10 --prefix 119 --gradient_accumulation_steps 8 --init_from_previous

