#!/data/user/yjiang717/bin/zsh
#SBATCH -J easyedit
#SBATCH -p acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -o slurm.log
#SBATCH -e slurm.log
#SBATCH --open-mode=append
#SBATCH --time=06:00:00

echo "\n\n[$(date '+%Y-%m-%d %H:%M:%S')] Job started.\n"

. ~/.zshrc
conda activate ke
# python run.py

cd examples
python run_wise_editing.py \
  --editing_method=WISE \
  --hparams_dir=../hparams/WISE/llama-7b \
  --data_dir=../data/wise \
  --ds_size=10 \
  --data_type=ZsRE \
  --sequential_edit

echo "\n[$(date '+%Y-%m-%d %H:%M:%S')] Job finished.\n\n"