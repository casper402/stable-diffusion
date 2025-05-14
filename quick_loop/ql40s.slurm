#!/bin/bash
#SBATCH --job-name=model_train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=202008583@post.au.dk
#SBATCH --partition=ql40s
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --export=NONE

cd $SLURM_SUBMIT_DIR

echo "========= Job started at $(date) =========="

echo "Job ID: $SLURM_JOB_ID"

module purge 
module load cuda/12.0.0
module load python/3.11.1

echo "Modules loaded"

source ../venvs/pytorch-l40s-env/bin/activate
which python
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
export PYTHONPATH=$PYTHONPATH:$(pwd)
python quick_loop/entry_casper.py

echo "========= Job finished at $(date) =========="
