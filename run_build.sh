#!/bin/bash
#SBATCH --job-name=build_u8
#SBATCH --output=/nas/longleaf/home/zixinl/logs/build_%j.out
#SBATCH --error=/nas/longleaf/home/zixinl/logs/build_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=general
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# No GPU needed — this is video decoding, which is CPU work.
# Running it on the general partition keeps your GPU allocation free.

cd /work/users/z/i/zixinl/Emotion
module load python/3.12.4
source .venv/bin/activate

echo "Started: $(date)"

python build_mmap.py --split val      --workers 8
python build_mmap.py --split test     --workers 8
python build_mmap.py --split trainval --workers 8

echo "Done: $(date)"
