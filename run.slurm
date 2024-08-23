#! /bin/bash
#SBATCH -A m4521
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -n 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task 32

module load python
module load gcc/10.3.0
module load cudatoolkit/11.5

MODEL_LOC='/pscratch/sd/l/lemonboy/alphaflow_develop/alphaflow'
DATA_LOC='/global/cfs/cdirs/m3513/metfish/PDB70_verB_fixed_data/'

srun /pscratch/sd/l/lemonboy/alphaflow_reinstall/bin/python $MODEL_LOC/train.py --lr 5e-4 --noise_prob 1 --accumulate_grad 8 --train_epoch_len 80 --pdb_chain "$DATA_LOC/result/msa/input_training.csv" --val_csv "$DATA_LOC/result/msa/input_validation.csv" --train_data_dir "$DATA_LOC/result" --train_msa_dir "$DATA_LOC/result/msa" --val_msa_dir "$DATA_LOC/result/msa" --saxs_dir "$DATA_LOC/result/saxs_r" --run_name "test_0814" --ckpt "/pscratch/sd/l/lemonboy/alphaflow_develop/alphaflow/workdir/test/epoch=5-step=6200.ckpt" --normal_validate --wandb

