#!/bin/bash
#SBATCH --partition=main
#SBATCH --requeue
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=8000
#SBATCH --cpus-per-task=1
#SBATCH --output=ARRARIDX.out
#SBATCH --error=ARRARIDX.err
 
cd $PWD

module purge

module load singularity

# module load slurm

PARAMS_FILE="$PWD/params_APT.txt"
read -r esC0 esC1 es0 es1  p_m L <<< $(sed -n "ARRARIDXp" $PARAMS_FILE)

echo $es0 $es1 $esC0 $esC1 $p_m $L
# srun singularity exec /scratch/hp636/python.sif mpirun -np 20 python -m mpi4py.futures run_APT_C_m.py -es $es0 $es1 -es_C $esC0 $esC1 -p_m $p_m $p_m 1 -p_f 1 1 1 -L $L
srun singularity exec /scratch/hp636/python.sif python -m mpi4py.futures run_APT_C_m_final.py -es_C $esC0 $esC1 -es $es0 $es1  -p_m $p_m $p_m 1 -p_f 1 1 1 -L $L

