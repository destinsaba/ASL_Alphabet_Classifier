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

wandb login 7239c9bb9691e04b4b97fd0cab3bc8d69e3d5cac

python /home/rana.elsadig/ASL_Alphabet_Classifier/classifier.py