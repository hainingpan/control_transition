#!/bin/bash
es=64
for n in 16 32 64; do
    ppn=$((64/n))
    for L in {26..12..-2}; do
        echo "n=$n, ppn=$ppn, L=$L, es=$es"
        mpirun -n $n -ppn $ppn python -m mpi4py.futures benchmark_cpu.py -L $L -xj "1/3,2/3" -es $es
    done
done