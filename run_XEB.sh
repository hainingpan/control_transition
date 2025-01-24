#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
##SBATCH --nodelist=gpu015,gpu016
#SBATCH --exclude=cuda00[1-8],gpuc00[1-2],pascal0[01-10],volta0[01-03],gpu0[05-14],gpu0[17-18]
#SBATCH --time=2:00:00
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

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

PARAMS_FILE="$PWD/params_XEB.txt"
read -r seed seed_C p_ctrl p_proj L1 L2 xj es_m es_C <<< $(sed -n "ARRARIDXp" $PARAMS_FILE)

echo $seed $seed_C $p_ctrl $p_proj $L1 $L2 $xj $es_m $es_C 

srun singularity exec --nv /scratch/hp636/pytorch.sif python run_XEB_encoding.py -seed $seed -seed_C $seed_C -p_ctrl $p_ctrl $p_ctrl 1 -p_proj $p_proj $p_proj 1 -L $L1 $L2 2 -xj "$xj" -es_m $es_m -es_C $es_C
