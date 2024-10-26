# mpirun -np 64 python -m mpi4py.futures run_APT.py -es 2000 -p_m 0.05 0.15 21 -p_f 1 1 1 -L 8
# mpirun -np 64 python -m mpi4py.futures run_APT.py -es 2000 -p_m 0.05 0.15 21 -p_f 1 1 1 -L 12 
mpirun -np 64 python -m mpi4py.futures run_APT.py -es 500 -p_m 0.05 0.15 21 -p_f 1 1 1 -L 16
# mpirun -np 64 python -m mpi4py.futures run_APT.py -es 2000 -p_m 0.05 0.15 21 -p_f 1 1 1 -L 20
# mpirun -np 64 python -m mpi4py.futures run_APT.py -es 2000 -p_m 0.05 0.15 21 -p_f 1 1 1 -L 24

# mpirun -np 64 python -m mpi4py.futures run_APT.py -es 2000 -p_m 0.05 0.35 21 -p_f 1 1 -1 -L 8
# mpirun -np 64 python -m mpi4py.futures run_APT.py -es 2000 -p_m 0.05 0.35 21 -p_f 1 1 -1 -L 12 
mpirun -np 64 python -m mpi4py.futures run_APT.py -es 500 -p_m 0.05 0.35 21 -p_f 1 1 -1 -L 16
# mpirun -np 64 python -m mpi4py.futures run_APT.py -es 2000 -p_m 0.05 0.35 21 -p_f 1 1 -1 -L 20
# mpirun -np 64 python -m mpi4py.futures run_APT.py -es 2000 -p_m 0.05 0.35 21 -p_f 1 1 -1 -L 24