#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

#SBATCH --exclude=cuda00[1-8],gpuc00[1-2],pascal0[01-10],volta0[01-03],gpu00[5-6],gpu0[17-18],gpu008,gpu0[10-16]
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8000
#SBATCH --cpus-per-task=1
#SBATCH --output=ARRARIDX.out
#SBATCH --error=ARRARIDX.err
##SBATCH --array=1-83%16
# Your commands here
 
cd $PWD

module purge

module load singularity

PARAMS_FILE="$PWD/params_AFM.txt"
# PARAMS_FILE="$PWD/params_0_anc_ent.txt"
# PARAMS_FILE="$PWD/params_1-3,2-3,-1-3.txt"
# read -r seed p_ctrl p_proj L1 L2 xj es ancilla <<< $(sed -n "${SLURM_ARRAY_TASK_ID}p" $PARAMS_FILE)
read -r seed p_ctrl p_proj L1 L2 xj es ancilla <<< $(sed -n "ARRARIDXp" $PARAMS_FILE)

echo $seed  $p_ctrl $p_proj $L1 $L2 "$xj" $es $ancilla
# for run_pytorch.py
srun singularity exec --nv /scratch/hp636/pytorch.sif python run_pytorch.py -seed $seed -p_ctrl $p_ctrl $p_ctrl 1 -p_proj $p_proj $p_proj 1 -L $L1 $L2 2 -xj "$xj" -es $es $ancilla

# # for run_pytorch_interpolation.py
# srun singularity exec --nv /scratch/hp636/pytorch.sif python run_pytorch_interpolation.py -seed $seed -p_ctrl $p_ctrl $p_ctrl 1 -p_global $p_global $p_global 1 -L $L1 $L2 2 -xj "$xj" -es $es $ancilla -p_proj 0. 0. 1

# for run_pytorch_sv.py, singular value
# srun singularity exec --nv /scratch/hp636/pytorch.sif python run_pytorch_sv.py -seed $seed -p_ctrl $p_ctrl $p_ctrl 1 -p_proj $p_proj $p_proj 1 -L $L1 $L2 2 -xj "$xj" -es $es $ancilla --complex128
