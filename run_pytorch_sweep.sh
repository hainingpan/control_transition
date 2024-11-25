#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=cuda00[1-8],gpuc00[1-2],pascal0[01-10],volta0[01-03],gpu0[05-14],gpu0[17-18]
#SBATCH --partition=main
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --mem=120000
#SBATCH --cpus-per-task=1
#SBATCH --output=ARRARIDX.out
#SBATCH --error=ARRARIDX.err
##SBATCH --array=1-83%16
# Your commands here
 
cd $PWD

module purge

module load singularity

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21

PARAMS_FILE="$PWD/params_0.txt"
# PARAMS_FILE="$PWD/params_1-3,2-3_anc_ctrl.txt"
# PARAMS_FILE="$PWD/params_0_ctrl.txt"
# PARAMS_FILE="$PWD/params_1-3,2-3,-1-3_2.txt"

# read -r seed p_ctrl p_proj L1 L2 xj es ancilla <<< $(sed -n "${SLURM_ARRAY_TASK_ID}p" $PARAMS_FILE)
read -r seed p_ctrl p_proj L1 L2 xj es ancilla <<< $(sed -n "ARRARIDXp" $PARAMS_FILE)
# read -r seed p_ctrl p_global L1 L2 xj es ancilla <<< $(sed -n "ARRARIDXp" $PARAMS_FILE)

echo $seed $p_ctrl $p_proj $L1 $L2 "$xj" $es $ancilla
# echo $seed  $p_ctrl $p_global $L1 $L2 "$xj" $es $ancilla
# for run_pytorch.py
# srun singularity exec --nv /scratch/hp636/pytorch.sif python run_pytorch.py -seed $seed -p_ctrl $p_ctrl $p_ctrl 1 -p_proj $p_proj $p_proj 1 -L $L1 $L2 2 -xj "$xj" -es $es $ancilla

# for run_pytorch_sv.py, singular value
# srun singularity exec --nv /scratch/hp636/pytorch.sif python run_pytorch_sv.py -seed $seed -p_ctrl $p_ctrl $p_ctrl 1 -p_proj $p_proj $p_proj 1 -L $L1 $L2 2 -xj "$xj" -es $es $ancilla --complex128

# # for run_pytorch.py
# srun singularity exec --nv /scratch/hp636/pytorch.sif python run_pytorch_interpolation.py -seed $seed -p_ctrl $p_ctrl $p_ctrl 1 -p_global $p_global $p_global 1 -L $L1 $L2 2 -xj "$xj" -es $es $ancilla -p_proj 0. 0. 1

# srun singularity exec --nv /scratch/hp636/pytorch.sif python run_pytorch.py -seed 1 -p_ctrl 0.60 0.60 1 -p_proj 0.48 0.48 1 -L 20 22 2 -xj "1/3,-1/3" -es 1000
# srun singularity exec --nv /scratch/hp636/pytorch.sif python run_pytorch.py -seed 0 -p_ctrl 0.42 0.42 1 -p_proj 0.00 0.00 1 -L 8 10 2 -xj "1/3,-1/3" -es 2000
# srun singularity exec --nv /scratch/hp636/pytorch.sif python test_random_seed.py

# for run_pytorch_sv.py, singular value
srun singularity exec --nv /scratch/hp636/pytorch.sif python run_pytorch_coherence_total.py -seed $seed -p_ctrl $p_ctrl $p_ctrl 1 -p_proj $p_proj $p_proj 1 -L $L1 $L2 2 -xj "$xj" -es $es $ancilla

# run_pytorch_coherence_total.py -seed 0 -p_ctrl 0.5 0.5 1 -p_proj 0 0 1 -L 10 12 2 -xj "0" -es 2 

