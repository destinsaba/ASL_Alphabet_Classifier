#! /bin/bash
# ============================
#SBATCH --job-name=ENEL645-proj
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=0-20:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --output=job_output.log
# ==================================

source ~/software/init-conda
conda activate pytorch

wandb login a384217417d4fb3d48a2b619be42f80bf3f7fd26

python /home/destin.saba/transfer-learning/ASLclassifier.py