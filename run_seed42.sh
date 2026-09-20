#!/bin/bash
#SBATCH --job-name=e2e_dev
#SBATCH --output=/nas/longleaf/home/zixinl/logs/e2e_%j.out
#SBATCH --error=/nas/longleaf/home/zixinl/logs/e2e_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

cd /work/users/z/i/zixinl/Emotion
module load python/3.12.4
module load cuda/12.6
source .venv/bin/activate

echo "Started: $(date)"

# Development run: train on Train rows, evaluate on Validation rows.
# Tune here.  Do NOT touch the test split until the configuration is frozen.
python train_e2e.py --mode dev --seed 42 --epochs 3

echo "Done: $(date)"
