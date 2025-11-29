#!/usr/bin/env python3
import numpy as np
import os

# Output directory where pickle files are saved
output_dir = os.path.expandvars('$WORKDIR/control_transition/APT_coherence_T')

# Coarser spacing for 0-0.08, finer spacing for 0.08-0.1
# p_m_values = np.hstack([np.arange(0, 0.08, 0.01), np.arange(0.08, 0.101, 0.005)])  # 13 values
# p_m_values = np.hstack([np.arange(0.11, 0.2, 0.01),])  # 13 values
p_m_values = np.hstack([np.arange(0.2, 1.0, 0.1),])  # 13 values

# Batching configuration (circuit ensemble size per job for different L values)
# Based on actual timing analysis from job 2089589 (extrapolated to 3h limit):
# L=20: can increase batch to 5987 → use 2000 (1 job)
# L=22: can increase batch to 1098 → use 1000 (2 jobs)
# L=24: can increase batch to 108 → use 100 (20 jobs)
# All values divide 2000 evenly for clean circuit coverage
batch_config = {
    12: {'es_C_batch': 2000, 'num_batches': 1},
    14: {'es_C_batch': 2000, 'num_batches': 1},
    16: {'es_C_batch': 2000, 'num_batches': 1},
    18: {'es_C_batch': 2000, 'num_batches': 1},
    20: {'es_C_batch': 2000, 'num_batches': 1},
    22: {'es_C_batch': 1000, 'num_batches': 2},
    24: {'es_C_batch': 100, 'num_batches': 20}
}

# Generate parameters for each L value separately, then combine
all_params = []

for L in [12, 14, 16, 18, 20, 22, 24]:
    es_C_batch = batch_config[L]['es_C_batch']
    num_batches = batch_config[L]['num_batches']
    es_start = 1
    es_end = 2  # Trajectory seed 1 only (arange(1,2) = [1])

    # Create circuit ensemble ranges for this L
    for batch_idx in range(num_batches):
        es_C_start = batch_idx * es_C_batch + 1  # Circuit seeds start at 1
        es_C_end = min((batch_idx + 1) * es_C_batch, 2000) + 1

        # For each p_m value, create a parameter line
        for p_m in p_m_values:
            # Filename: APT_En(es_start,es_end)_EnC(es_C_start,es_C_end)_pm(p_m,p_m,1)_pf(0,0,-1)_L{L}_coherence_T.pickle
            filename = f'APT_En({es_start},{es_end})_EnC({es_C_start},{es_C_end})_pm({p_m:.3f},{p_m:.3f},1)_pf(0.000,0.000,-1)_L{L}_coherence_T.pickle'

            # Parameter line for submit script
            param_line = f'--L {L} --p_m {p_m:.3f} {p_m:.3f} 1 --es {es_start} {es_end} --es_C {es_C_start} {es_C_end}'

            all_params.append(param_line)

# Write all parameters to a single file
filename = 'params_APT_coherence_T_2.txt'
with open(filename, 'w') as f:
    for param in all_params:
        f.write(param + '\n')

print(f"Generated {filename} with {len(all_params)} lines")
