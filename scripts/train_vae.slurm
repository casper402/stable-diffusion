#!/bin/bash
#SBATCH --job-name=model_train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=202008583@post.au.dk
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=5
#SBATCH --time=12:00:00
#SBATCH --export=NONE

cd $SLURM_SUBMIT_DIR

echo "========= Job started at $(date) =========="

echo "Job ID: $SLURM_JOB_ID"

module purge
module load anaconda3/2021.05 
module load gcc/9.3.0
module load cuda/11.2.0

source activate cuda_env

export PYTHONPATH=$PYTHONPATH:$(pwd)
python vae.py

echo "========= Job finished at $(date) =========="