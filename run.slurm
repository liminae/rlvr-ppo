#!/bin/bash
#SBATCH --job-name=ppo_gsm8k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=ppo_%j.out

module load cuda
source ~/assip/env/bin/activate
cd ~/assip

echo "Starting PPO fine‑tune on $(hostname) at $(date)"
python3 ppo.py
echo "Finished at $(date)"
