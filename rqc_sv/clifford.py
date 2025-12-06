import numpy as np
import stim
import galois

GF2 = galois.GF(2)


class Clifford:
    """Flagged Clifford Circuit (A_1^alpha Protocol) using stim stabilizer simulation."""

    def __init__(self, L, seed=None, seed_C=None, store_op=False, alpha=2.0):
        self.L = L
        self.store_op = store_op
        self.alpha = alpha

        self.rng = np.random.default_rng(seed)
        self.rng_C = np.random.default_rng(seed_C)

        # Precompute pair indices with periodic BC
        self.odd_pairs = [(i, i+1) for i in range(0, L, 2)]
        self.even_pairs = [(i, (i+1) % L) for i in range(1, L, 2)]

        # Cache all 11520 2-qubit Cliffords
        self.clifford_2q_cache = list(stim.Tableau.iter_all(2))

        # Pre-cache Z_i Z_j PauliStrings for OP2
        self._zz_paulis = {}
        for i in range(L):
            for j in range(i + 1, L):
                p = stim.PauliString(L)
                p[i] = 3  # Z
                p[j] = 3  # Z
                self._zz_paulis[(i, j)] = p

        self.op_history = []
        self.sim = stim.TableauSimulator()
        self._initialize_state()

    def _initialize_state(self):
        """Initialize simulator, qubits to |1⟩ (defect), and flags to 0."""
        self.sim.reset(*range(self.L))
        for i in range(self.L):
            self.sim.x(i)
        self.flags = np.zeros(self.L, dtype=int)

    def _measure_z_seeded(self, qubit):
        """Measure Z with outcome controlled by self.rng. Returns 0 for |0⟩, 1 for |1⟩."""
        z_exp = self.sim.peek_z(qubit)
        if z_exp == 0:
            outcome = int(self.rng.random() >= 0.5)
        else:
            outcome = 0 if z_exp == 1 else 1
        self.sim.postselect_z(qubit, desired_value=bool(outcome))
        return outcome

    def measure_pair(self, i, j):
        """MEASURE_PAIR: Measure Z on qubits i,j and update flags."""
        outcome_i = self._measure_z_seeded(i)
        outcome_j = self._measure_z_seeded(j)
        self.flags[i] = 1 - outcome_i
        self.flags[j] = 1 - outcome_j
        if self.store_op:
            self.op_history.append(('measure_pair', i, j, 1-2*outcome_i, 1-2*outcome_j))

    def _apply_random_clifford_2q(self, i, j):
        """Apply uniformly random 2-qubit Clifford to qubits i,j."""
        idx = self.rng_C.integers(0, len(self.clifford_2q_cache))
        self.sim.do_tableau(self.clifford_2q_cache[idx], [i, j])

    def conditional_unitary(self, i, j):
        """CONDITIONAL_UNITARY: Identity if both flags=1, else random Clifford and reset flags."""
        if self.flags[i] == 1 and self.flags[j] == 1:
            if self.store_op:
                self.op_history.append(('identity', i, j))
        else:
            self._apply_random_clifford_2q(i, j)
            self.flags[i] = 0
            self.flags[j] = 0
            if self.store_op:
                self.op_history.append(('clifford', i, j))

    def _sample_power_law_pairs(self, defect_indices, n_pairs):
        """Sample n_pairs pairs from defects with probability ∝ dist^(-alpha)."""
        defect_indices = np.asarray(defect_indices)
        n = len(defect_indices)
        if n < 2 or n_pairs == 0:
            return []

        # Periodic boundary: r = min(|i-j|, L-|i-j|)
        raw_dist = np.abs(defect_indices[:, None] - defect_indices[None, :])
        dist_matrix = np.minimum(raw_dist, self.L - raw_dist)
        triu_i, triu_j = np.triu_indices(n, k=1)
        weights = dist_matrix[triu_i, triu_j] ** (-self.alpha)
        weights /= weights.sum()

        pair_indices = self.rng_C.choice(len(weights), size=n_pairs, p=weights)
        return [(defect_indices[triu_i[idx]], defect_indices[triu_j[idx]]) for idx in pair_indices]

    def random_circuit(self, p_m):
        """Execute one timestep with 5 layers."""
        # Layer 1: Odd Measurements
        for i, j in self.odd_pairs:
            if self.rng_C.random() < p_m:
                self.measure_pair(i, j)

        # Layer 2: Odd Unitaries
        for i, j in self.odd_pairs:
            self.conditional_unitary(i, j)

        # Layer 3: Even Measurements
        for i, j in self.even_pairs:
            if self.rng_C.random() < p_m:
                self.measure_pair(i, j)

        # Layer 4: Even Unitaries
        for i, j in self.even_pairs:
            self.conditional_unitary(i, j)

        # Layer 5: Variable-range Control
        defect_indices = np.where(self.flags == 0)[0]
        n_def = len(defect_indices)
        if n_def >= 2:
            pairs = self._sample_power_law_pairs(defect_indices, n_def)
            for u, v in pairs:
                self._apply_random_clifford_2q(u, v)
                if self.store_op:
                    self.op_history.append(('variable_range_clifford', u, v))

    def OP(self):
        """Order parameter (density of spin up): (1/L) sum_i (1+Z_i)/2."""
        z_expectations = np.array([self.sim.peek_z(i) for i in range(self.L)])
        return np.mean((1 + z_expectations) / 2)

    def OP2(self):
        """<OP^2> = (1/4L^2) [L^2 + 2L*sum_i<Z_i> + sum_{i,j}<Z_i Z_j>]. Uses pre-cached PauliStrings."""
        z_exp = np.array([self.sim.peek_z(i) for i in range(self.L)])
        sum_z = np.sum(z_exp)

        # sum_{i,j} <Z_i Z_j> (includes diagonal Z_i^2 = 1)
        zz_total = float(self.L)  # Diagonal contribution
        for (i, j), pauli in self._zz_paulis.items():
            zz_total += 2 * self.sim.peek_observable_expectation(pauli)

        return (self.L**2 + 2*self.L*sum_z + zz_total) / (4 * self.L**2)

    def OP2_naive(self):
        """<OP^2> = (1/4L^2) [L^2 + 2L*sum_i<Z_i> + sum_{i,j}<Z_i Z_j>]. Creates PauliStrings each call."""
        z_exp = np.array([self.sim.peek_z(i) for i in range(self.L)])
        sum_z = np.sum(z_exp)

        # sum_{i,j} <Z_i Z_j>
        zz_total = float(self.L)  # Diagonal
        for i in range(self.L):
            for j in range(i + 1, self.L):
                pauli = stim.PauliString(self.L)
                pauli[i] = 3
                pauli[j] = 3
                zz_total += 2 * self.sim.peek_observable_expectation(pauli)

        return (self.L**2 + 2*self.L*sum_z + zz_total) / (4 * self.L**2)

    def OP2_sparse(self):
        """<OP^2> using sparse computation - O(L + k^2) where k is # uncertain qubits.

        Exploits that for stabilizer states:
        - If peek_z(i) = ±1, qubit i has definite Z value and correlations factorize
        - <Z_i Z_j> = <Z_i><Z_j> when both are definite
        - <Z_i Z_j> = 0 when i is definite and j is uncertain (since <Z_j> = 0)
        - Only uncertain-uncertain pairs need explicit computation
        """
        # O(L) to get single-qubit Z expectations
        z_exp = np.array([self.sim.peek_z(i) for i in range(self.L)])
        sum_z = np.sum(z_exp)

        uncertain = np.where(z_exp == 0)[0]
        definite = np.where(z_exp != 0)[0]

        # Diagonal contribution: sum_i <Z_i^2> = L
        zz_total = float(self.L)

        # Definite-definite pairs: <Z_i Z_j> = z_exp[i] * z_exp[j]
        # sum_{i<j} 2*z_exp[i]*z_exp[j] = (sum z_exp[definite])^2 - n_def
        if len(definite) > 0:
            def_sum = np.sum(z_exp[definite])
            zz_total += def_sum**2 - len(definite)

        # Definite-uncertain pairs: <Z_i Z_j> = z_exp[i] * <Z_j> = z_exp[i] * 0 = 0
        # (no contribution)

        # Uncertain-uncertain pairs: compute explicitly using cached PauliStrings
        for idx_i in range(len(uncertain)):
            for idx_j in range(idx_i + 1, len(uncertain)):
                qi, qj = uncertain[idx_i], uncertain[idx_j]
                zz_total += 2 * self.sim.peek_observable_expectation(self._zz_paulis[(qi, qj)])

        return (self.L**2 + 2*self.L*sum_z + zz_total) / (4 * self.L**2)

    def OP2_adaptive(self, p_m, p_m_critical=0.3):
        """Adaptive <OP^2> that switches method based on measurement rate.

        Uses OP2_sparse() when p_m > p_m_critical (absorbing phase, few uncertain qubits).
        Uses OP2() when p_m <= p_m_critical (active phase, many uncertain qubits).

        Parameters
        ----------
        p_m : float
            Current measurement probability.
        p_m_critical : float, optional
            Critical measurement probability for method switching, by default 0.3.

        Returns
        -------
        float
            <OP^2> value.
        """
        if p_m > p_m_critical:
            return self.OP2_sparse()
        else:
            return self.OP2()

    def get_tableau(self):
        """Get stabilizer tableau as (x_sector, z_sector) boolean arrays."""
        tab = self.sim.current_inverse_tableau().inverse()
        _, _, z2x, z2z, _, _ = tab.to_numpy()
        return z2x, z2z

    def _stabilizer_entropy(self, subregion):
        """Entanglement entropy S(A) = |A| - rank(stabilizers on A)."""
        x_sec, z_sec = self.get_tableau()
        combined = np.logical_or(x_sec[:, subregion], z_sec[:, subregion])
        return len(subregion) - np.linalg.matrix_rank(GF2(combined.astype(int)))

    def half_system_entanglement_entropy(self, selfaverage=False):
        """Half-system entanglement entropy."""
        if selfaverage:
            return np.mean([self._stabilizer_entropy(list(range(i, i + self.L // 2)))
                           for i in range(self.L // 2)])
        return self._stabilizer_entropy(list(range(self.L // 2)))

    def von_Neumann_entropy_pure(self, subregion):
        """Von Neumann entropy for a subregion."""
        return self._stabilizer_entropy(list(subregion))

    def quantum_L1_coherence(self):
        """Quantum coherence: rank of X-sector."""
        x_sec, _ = self.get_tableau()
        return np.linalg.matrix_rank(GF2(x_sec.astype(int)))

    def tripartite_mutual_information(self, subregion_A, subregion_B, subregion_C, selfaverage=False):
        """Tripartite mutual information I_3."""
        subregion_A = np.asarray(subregion_A)
        subregion_B = np.asarray(subregion_B)
        subregion_C = np.asarray(subregion_C)

        if selfaverage:
            return np.mean([
                self.tripartite_mutual_information(
                    (subregion_A + shift) % self.L,
                    (subregion_B + shift) % self.L,
                    (subregion_C + shift) % self.L
                ) for shift in range(len(subregion_A))
            ])

        S = self._stabilizer_entropy
        return (S(list(subregion_A)) + S(list(subregion_B)) + S(list(subregion_C))
                - S(list(np.concatenate([subregion_A, subregion_B])))
                - S(list(np.concatenate([subregion_A, subregion_C])))
                - S(list(np.concatenate([subregion_B, subregion_C])))
                + S(list(np.concatenate([subregion_A, subregion_B, subregion_C]))))
