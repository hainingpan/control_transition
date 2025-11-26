#!/usr/bin/env python3
from rqc import generate_params, load_pickle
import numpy as np
import os

# Output directory where pickle files are saved
output_dir = os.path.expandvars('$WORKDIR/control_transition/APT_coherence_T')

# Coarser spacing for 0-0.08, finer spacing for 0.08-0.1
p_m_values = np.hstack([np.arange(0, 0.08, 0.01), np.arange(0.08, 0.101, 0.005)])  # 13 values

# Batching configuration (ensemble size per job for different L values)
batch_config = {
    12: {'es_batch': 2000, 'num_batches': 1},
    14: {'es_batch': 2000, 'num_batches': 1},
    16: {'es_batch': 2000, 'num_batches': 1},
    18: {'es_batch': 1000, 'num_batches': 2},
    20: {'es_batch': 250, 'num_batches': 8},
    22: {'es_batch': 50, 'num_batches': 40},
    24: {'es_batch': 10, 'num_batches': 200}
}

# Generate parameters for each L value separately, then combine
all_params = []

for L in [12, 14, 16, 18, 20, 22, 24]:
    es_batch = batch_config[L]['es_batch']
    num_batches = batch_config[L]['num_batches']

    # Create ensemble ranges for this L
    es_ranges = []
    for batch_idx in range(num_batches):
        es_start = batch_idx * es_batch
        es_end = min((batch_idx + 1) * es_batch, 2000)
        es_ranges.append((es_start, es_end))

    # Generate parameters for this L using rqc.generate_params
    params_list = [
        ({'es_C_start': 1, 'es_C_end': 2, 'p_f': -1, 'L': L},
         {
             'p_m': list(p_m_values),
             'es_range': es_ranges
         }
        ),
    ]

    for fixed_params, vary_params in params_list:
        params = generate_params(
            fixed_params=fixed_params,
            vary_params=vary_params,
            fn_template='APT_En({es_range[0]},{es_range[1]})_EnC({es_C_start},{es_C_end})_pm({p_m:.3f},{p_m:.3f},1)_pf(0.000,0.000,{p_f})_L{L}_coherence_T.pickle',
            fn_dir_template=output_dir,
            input_params_template='{L} {p_m:.3f} {es_range[0]} {es_range[1]}',
            load_data=load_pickle,
            filename=f'params_APT_coherence_T_L{L}.txt',
            filelist=None,
            load=False,
            data_dict=None,
        )
        all_params.extend(params)

# Combine all parameters into a single file
with open('params_APT_coherence_T.txt', 'w') as f:
    for param in all_params:
        f.write(param + '\n')

print(f"Generated params_APT_coherence_T.txt with {len(all_params)} lines")
