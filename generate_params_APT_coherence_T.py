#!/usr/bin/env python3
import numpy as np
import os
from rqc import generate_params

# Output directory where pickle files are saved
# output_dir = os.path.expandvars('$WORKDIR/control_transition/APT_coherence_T')
output_dir = os.path.join(os.environ.get('WORKDIR', '..'), 'control_transition/APT_coherence_T')

# Tunable parameter: p_m values sweep
# p_m_values = np.hstack([np.arange(0, 0.08, 0.01), np.arange(0.08, 0.101, 0.005),np.arange(0.11, 0.2, 0.01), np.arange(0.2, 0.35, 0.02)]) # this is p_m = p_f 
# p_m_values = np.hstack([np.arange(0, 0.06, 0.02), np.arange(0.06, 0.08, 0.01), np.arange(0.085, 0.101, 0.005),np.arange(0.11, 0.13, 0.01), ])  # Coarse/fine spacing
p_m_values = np.hstack([np.arange(0.085, 0.101, 0.005),np.arange(0.11, 0.13, 0.01), ])  # Coarse/fine spacing

# pf = [0.0, 0.0, -1]
pf = [1,1,1]

output_filename = 'params_APT_coherence_T.txt'
# output_filename = 'params_APT_coherence_T_2.txt'


# Tunable parameter: L values


# Tunable parameter: Batching configuration (circuit ensemble size per job for different L values)
# Based on actual timing analysis from job 2089589 (extrapolated to 3h limit):
# L=20: can increase batch to 5987 → use 2000 (1 job)
# L=22: can increase batch to 1098 → use 1000 (2 jobs)
# L=24: can increase batch to 108 → use 100 (20 jobs)
# All values divide 2000 evenly for clean circuit coverage

# This is for base ~2000 circuits  
# batch_config = {
#     12: {'es_C_batch': 2000, 'num_batches': 1},
#     14: {'es_C_batch': 2000, 'num_batches': 1},
#     16: {'es_C_batch': 2000, 'num_batches': 1},
#     18: {'es_C_batch': 2000, 'num_batches': 1},
#     20: {'es_C_batch': 1000, 'num_batches': 2},
#     22: {'es_C_batch': 24*10, 'num_batches': 2000//(24*10)+1},
#     24: {'es_C_batch': 24*2, 'num_batches': 2000//(24*2)+1}
# }
# This is for base ~4000 circuits
batch_config = {
    12: {'es_C_batch': 2000, 'num_batches': 2},
    14: {'es_C_batch': 2000, 'num_batches': 2},
    16: {'es_C_batch': 2000, 'num_batches': 2},
    18: {'es_C_batch': 2000, 'num_batches': 2},
    20: {'es_C_batch': 1000, 'num_batches': 4},
    22: {'es_C_batch': 24*10, 'num_batches': 4000//(24*10)+1},
    24: {'es_C_batch': 24*2, 'num_batches': 4000//(24*2)+1}
}
L_values = list(batch_config.keys())

# Tunable parameter: Trajectory seed range
es_start = 1
es_end = 2  # Trajectory seed 1 only (arange(1,2) = [1])

# Build params_list for each (L, batch) combination
params_list = []

for L in L_values:
    es_C_batch = batch_config[L]['es_C_batch']
    num_batches = batch_config[L]['num_batches']

    for batch_idx in range(num_batches):
        es_C_start = batch_idx * es_C_batch + 1  # Circuit seeds start at 1
        es_C_end = (batch_idx + 1) * es_C_batch+ 1

        fixed_params = {
            'L': L,
            'es_start': es_start,
            'es_end': es_end,
            'es_C_start': es_C_start,
            'es_C_end': es_C_end,
            'pf1': pf[0],
            'pf2': pf[1],
            'pf3': pf[2],
        }

        vary_params = {
            'p_m': p_m_values,
        }

        params_list.append((fixed_params, vary_params))

# Output filename for parameters

# Generate parameters for each (L, batch) combination
for fixed_params, vary_params in params_list:
    generate_params(
        fixed_params=fixed_params,
        vary_params=vary_params,
        fn_template='APT_En({es_start},{es_end})_EnC({es_C_start},{es_C_end})_pm({p_m:.3f},{p_m:.3f},1)_pf({pf1:.3f},{pf2:.3f},{pf3:d})_L{L}_coherence_T.pickle',
        fn_dir_template='APT_coherence_T_pf1',
        input_params_template='--L {L} --p_m {p_m:.3f} {p_m:.3f} 1 --p_f {pf1:.3f} {pf2:.3f} {pf3:d} --es {es_start} {es_end} --es_C {es_C_start} {es_C_end}',
        load_data=lambda x: None,
        filename=output_filename,
        load=False,
        data_dict=None,
    )

print(f"Generated {output_filename}")
