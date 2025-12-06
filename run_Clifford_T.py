from rqc_sv.clifford import Clifford
import numpy as np
from joblib import Parallel, delayed
from time import time
import argparse
import pickle
from tqdm import tqdm
import os


def run(inputs):
    L, p_m, alpha, seed, seed_C = inputs
    cliff = Clifford(L=L, seed=seed, seed_C=seed_C, store_op=False, alpha=alpha)
    tf = int(L**1.62)
    OP_list = []
    OP2_list = []

    # OP_list.append(cliff.OP())
    # OP2_list.append(cliff.OP2_adaptive(p_m))
    for i in range(tf):
        cliff.random_circuit(p_m=p_m)
        # OP_list.append(cliff.OP())
        # OP2_list.append(cliff.OP2_adaptive(p_m))
    OP_list.append(cliff.OP())
    OP2_list.append(cliff.OP2_adaptive(p_m))
    return OP_list, OP2_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', '-L', type=int, default=12, help='System size (default: 12).')
    parser.add_argument('--es', '-es', default=[1, 10], type=int, nargs=2, help='Ensemble size range [start, end) (default: [1, 10]).')
    parser.add_argument('--es_C', '-es_C', default=[1, 10], type=int, nargs=2, help='Ensemble size for circuit [start, end) (default: [1, 10]).')
    parser.add_argument('--p_m', '-p_m', type=float, nargs=3, default=[0, 1, 11], help='Parameters for p_m in the form [start, stop, num] (default: [0, 1, 11]).')
    parser.add_argument('--alpha', '-a', type=float, default=2.0, help='Power-law exponent for variable-range control (default: 2.0).')
    parser.add_argument('--n_jobs', '-n', type=int, default=-1, help='Number of parallel jobs (-1 for all available cores, default: -1).')
    args = parser.parse_args()

    p_m_list = np.linspace(args.p_m[0], args.p_m[1], int(args.p_m[2]))
    es_list = np.arange(args.es[0], args.es[1])
    es_C_list = np.arange(args.es_C[0], args.es_C[1])

    inputs = [(args.L, p_m, args.alpha, idx, idx_C)
              for p_m in p_m_list
              for idx in es_list
              for idx_C in es_C_list]

    st = time()

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(run)(input_data) for input_data in tqdm(inputs, desc="Processing")
    )

    # Separate OP and OP2 results
    OP_results = [r[0] for r in results]
    OP2_results = [r[1] for r in results]

    OP_map = np.array(OP_results).reshape((p_m_list.shape[0], es_list.shape[0], es_C_list.shape[0], -1))
    OP2_map = np.array(OP2_results).reshape((p_m_list.shape[0], es_list.shape[0], es_C_list.shape[0], -1))

    output_dir = os.environ.get('WORKDIR', '..')
    filename = f'{output_dir}/control_transition/Clifford_En({args.es[0]:d},{args.es[1]:d})_EnC({args.es_C[0]:d},{args.es_C[1]:d})_pm({args.p_m[0]:.3f},{args.p_m[1]:.3f},{args.p_m[2]:.0f})_alpha{args.alpha:.1f}_L{args.L:d}_T.pickle'

    with open(filename, 'wb') as f:
        pickle.dump({"OP": OP_map, "OP2": OP2_map, "args": args}, f)

    print(f'Saved to: {filename}')
    print(f'Time elapsed: {time()-st:.4f}s')

# OMP_NUM_THREADS=1 python run_Clifford_T.py -L 128 -es 1 2 -es_C 1 50 -p_m 0.5 .9 11 -a 3.5 -n -1