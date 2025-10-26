#!/data/user/yjiang717/bin/zsh
#SBATCH -J easyedit
#SBATCH -p acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -o slurm.log
#SBATCH -e slurm.log
#SBATCH --open-mode=append
#SBATCH --time=00:15:00

. ~/.zshrc
# conda activate easyedit
conda activate ke
python run.py
