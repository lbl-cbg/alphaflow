#! /bin/bash
#SBATCH -A m4521
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 6:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task 32

module load python
module load gcc/10.3.0
module load cudatoolkit/11.5

MODEL_LOC='/pscratch/sd/l/lemonboy/alphaflow_develop/alphaflow'

srun /pscratch/sd/l/lemonboy/alphaflow_reinstall/bin/python $MODEL_LOC/predict.py --input_csv "/pscratch/sd/l/lemonboy/alphaflow_develop/test_dataset/input_no_training_data.csv" --msa_dir  "/global/cfs/cdirs/m3513/metfish/apo_holo_data/msa" --saxs_dir "/global/cfs/cdirs/m3513/metfish/apo_holo_data/saxs_r_feng" --ckpt "/pscratch/sd/l/lemonboy/alphaflow_develop/alphaflow/workdir/test/epoch=5-step=6200.ckpt" --samples "10" --outpdb "/pscratch/sd/l/lemonboy/alphaflow_develop/test_dataset" --resample 
