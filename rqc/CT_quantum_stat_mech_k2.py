import numpy as np

class CT_quantum_stat_mech_k2:
    """k=2 statistical mechanics model for second moment calculations.

    State representation per site:
    - 0 = U (up/reset): both copies in |0><0|
    - 1 = I (identity): maximally mixed on both copies
    - 2 = S (swap): swap operator between copies
    """

    def __init__(self, L, seed=None, seed_vec=None, seed_C=None, x0=None):
        """Initialize the k=2 statistical mechanics model.

        Parameters:
        -----------
        L : int
            System size (number of sites)
        seed : int, optional
            Random seed for general RNG
        seed_vec : int, optional
            Random seed for vector initialization (currently unused)
        seed_C : int, optional
            Random seed for circuit operations
        x0 : array-like, optional
            Initial state (currently unused, always starts as all U)
        """
        self.L = L
        self.rng = np.random.default_rng(seed)
        self.rng_C = np.random.default_rng(seed_C) if seed_C is not None else self.rng
        self.x0 = x0

        # State vector: 0=U, 1=I, 2=S
        self.vec = self._initialize_vector()

        # Log-weight to avoid numerical underflow
        # Actual weight w = exp(log_weight)
        self.log_weight = 0.0

        # Pre-compute log values for weight updates
        self.log_2_5 = np.log(2.0/5.0)  # U-U chaotic
        self.log_3_5 = np.log(3.0/5.0)  # U-I or U-S chaotic
        self.log_4_5 = np.log(4.0/5.0)  # I-S or S-I chaotic
        self.log_2 = np.log(2.0)         # Control on I

    def _initialize_vector(self):
        """Initialize state vector to all U (all sites reset)."""
        return np.zeros(self.L, dtype=int)

    def Bernouli(self):
        """Apply chaotic two-qubit gate on the bond (last two sites).

        Rules based on bond state:
        - U-U (0,0): Choose II or SS with prob 1/2 each, w *= 2/5
        - U-I (0,1) or I-U (1,0): Choose II (7/9) or SS (2/9), w *= 3/5
        - U-S (0,2) or S-U (2,0): Choose II (2/9) or SS (7/9), w *= 3/5
        - I-I (1,1) or S-S (2,2): No change
        - I-S (1,2) or S-I (2,1): Choose II or SS with prob 1/2, w *= 4/5

        Then roll the vector left by 1.
        """
        # Get the bond state (last two positions)
        j, jp = self.vec[-2], self.vec[-1]

        if j == 0 and jp == 0:  # U-U
            # Choose II or SS with prob 1/2 each
            if self.rng.random() < 0.5:
                self.vec[-2:] = 1  # II
            else:
                self.vec[-2:] = 2  # SS
            self.log_weight += self.log_2_5

        elif (j == 0 and jp == 1) or (j == 1 and jp == 0):  # U-I or I-U
            # Choose II with prob 7/9, SS with prob 2/9
            if self.rng.random() < 7.0/9.0:
                self.vec[-2:] = 1  # II
            else:
                self.vec[-2:] = 2  # SS
            self.log_weight += self.log_3_5

        elif (j == 0 and jp == 2) or (j == 2 and jp == 0):  # U-S or S-U
            # Choose II with prob 2/9, SS with prob 7/9
            if self.rng.random() < 2.0/9.0:
                self.vec[-2:] = 1  # II
            else:
                self.vec[-2:] = 2  # SS
            self.log_weight += self.log_3_5

        elif (j == 1 and jp == 1) or (j == 2 and jp == 2):  # I-I or S-S
            # No change to state or weight
            pass

        elif (j == 1 and jp == 2) or (j == 2 and jp == 1):  # I-S or S-I
            # Choose II or SS with prob 1/2 each
            if self.rng.random() < 0.5:
                self.vec[-2:] = 1  # II
            else:
                self.vec[-2:] = 2  # SS
            self.log_weight += self.log_4_5

        # Roll vector left by 1
        self.vec = np.roll(self.vec, -1)

    def control(self):
        """Apply control (reset) operation on the last site.

        - Set last site to U (0)
        - Update weight: if previous state was I, multiply w by 2
        - Roll vector right by 1
        """
        # Store previous state
        prev_state = self.vec[-1]

        # Set to U
        self.vec[-1] = 0

        # Update weight if previous was I
        if prev_state == 1:  # Was I
            self.log_weight += self.log_2
        # If was S (2) or U (0): no weight change

        # Roll vector right by 1
        self.vec = np.roll(self.vec, 1)

    def random_circuit(self, p):
        """Apply random circuit with control probability p.

        Parameters:
        -----------
        p : float
            Probability of applying control (vs chaotic gate)
        """
        prob = self.rng_C.random()
        if prob >= p:
            self.Bernouli()
        else:
            self.control()

    def get_weight(self):
        """Return the actual trajectory weight w = exp(log_weight)."""
        return np.exp(self.log_weight)

    def get_log_weight(self):
        """Return the log of trajectory weight."""
        return self.log_weight

    def mean_Mz_moment(self, n=1):
        """Compute the n-th moment of Mz for the k=2 model.

        For k=2, we compute Tr(rho^(2) (Mz tensor Mz)).
        The contribution from each site to <Z_A * Z_B>:
        - U (0): contributes 1 (both copies are |0><0|)
        - I (1): contributes 0 (maximally mixed)
        - S (2): contributes 1 (swap of two aligned states)

        For n=2: Tr(rho^(2) (Mz tensor Mz)) = (N_U/L)^2 + N_S/L^2

        Parameters:
        -----------
        n : int
            Moment order (default=1)

        Returns:
        --------
        float
            The n-th moment of Mz tensor Mz
        """
        N_U = np.sum(self.vec == 0)
        N_I = np.sum(self.vec == 1)
        N_S = np.sum(self.vec == 2)

        if n == 1:
            # For n=1: Tr(rho^(2) (Mz ⊗ Mz)) where Mz = (1/L) sum_i Z_i
            #
            # Using the per-site joint probabilities p_i^{ab}:
            # - U: a_i = b_i = +1, c_i = +1
            # - I: a_i = b_i = 0, c_i = 0
            # - S: a_i = b_i = 0, c_i = +1
            # where a_i = E[Z_i^A], b_i = E[Z_i^B], c_i = E[Z_i^A Z_i^B]
            #
            # Formula: Tr(rho^(2) (Mz⊗Mz)) = (1/L^2)[(sum_i a_i)(sum_j b_j) + sum_i(c_i - a_i*b_i)]
            #        = (1/L^2)[N_U * N_U + N_S] = (N_U^2 + N_S) / L^2
            return (N_U**2 + N_S) / self.L**2

        elif n == 2:
            # For n=2: (Tr(rho^(2) (Mz tensor Mz)))^2
            val_1 = (N_U**2 + N_S) / self.L**2
            return val_1**2

        else:
            # General case
            val_1 = (N_U**2 + N_S) / self.L**2
            return val_1**n

    def _compute_joint_probs(self):
        """Compute joint probabilities p_i^{ab} for each site.

        Returns:
        --------
        p_joint : ndarray, shape (L, 2, 2)
            p_joint[i, a, b] = p_i^{ab} = Pr(bit_i^A = a, bit_i^B = b)
            where a, b in {0, 1}
        """
        p_joint = np.zeros((self.L, 2, 2))

        for i in range(self.L):
            state = self.vec[i]
            if state == 0:  # U
                p_joint[i, 0, 0] = 1.0
            elif state == 1:  # I
                p_joint[i, :, :] = 0.25
            elif state == 2:  # S
                p_joint[i, 0, 0] = 0.5
                p_joint[i, 1, 1] = 0.5

        return p_joint

    def _compute_marginals(self, p_joint):
        """Compute marginal probabilities from joint probabilities.

        Parameters:
        -----------
        p_joint : ndarray, shape (L, 2, 2)
            Joint probabilities

        Returns:
        --------
        P_A : ndarray, shape (L, 2)
            P_A[i, a] = P_i^A(a) = marginal probability for replica A
        P_B : ndarray, shape (L, 2)
            P_B[i, b] = P_i^B(b) = marginal probability for replica B
        """
        # P_i^A(0) = p_i^{00} + p_i^{01}
        # P_i^A(1) = p_i^{10} + p_i^{11}
        P_A = np.sum(p_joint, axis=2)

        # P_i^B(0) = p_i^{00} + p_i^{10}
        # P_i^B(1) = p_i^{01} + p_i^{11}
        P_B = np.sum(p_joint, axis=1)

        return P_A, P_B

    def mean_FDW_moment(self, n=1):
        """Compute the n-th moment of FDW for the k=2 model.

        Uses the explicit formula for Tr(rho^(2) ((F tensor F)^n))
        where F is the First Domain Wall operator.

        Parameters:
        -----------
        n : int
            Moment order (default=1)

        Returns:
        --------
        float
            The n-th moment of F tensor F
        """
        # Compute joint probabilities and marginals
        p_joint = self._compute_joint_probs()
        P_A, P_B = self._compute_marginals(p_joint)

        # Note: positions are indexed from right, so position 1 is rightmost (i=0)
        # and position L is leftmost (i=L-1)
        # We'll work with 1-based indexing for positions as in the spec

        result = 0.0

        # Sum over all pairs (k, l) where k, l are positions from right
        for k in range(1, self.L + 1):
            for l in range(1, self.L + 1):
                # Compute Pr(F_A = k, F_B = l)

                if k < l:
                    # Case k < l
                    # Pr = [prod_{i=1}^{k-1} p_i^{00}] * p_k^{10} *
                    #      [prod_{i=k+1}^{l-1} P_i^B(0)] * P_l^B(1)

                    prob = 1.0

                    # Product over i=1 to k-1 (positions from right)
                    # Position i from right corresponds to index (i-1) in arrays
                    for i in range(1, k):
                        idx = i - 1
                        prob *= p_joint[idx, 0, 0]

                    # p_k^{10}
                    prob *= p_joint[k-1, 1, 0]

                    # Product over i=k+1 to l-1
                    for i in range(k+1, l):
                        idx = i - 1
                        prob *= P_B[idx, 0]

                    # P_l^B(1)
                    prob *= P_B[l-1, 1]

                elif k == l:
                    # Case k = l
                    # Pr = [prod_{i=1}^{k-1} p_i^{00}] * p_k^{11}

                    prob = 1.0

                    # Product over i=1 to k-1
                    for i in range(1, k):
                        idx = i - 1
                        prob *= p_joint[idx, 0, 0]

                    # p_k^{11}
                    prob *= p_joint[k-1, 1, 1]

                else:  # l < k
                    # Case l < k
                    # Pr = [prod_{i=1}^{l-1} p_i^{00}] * p_l^{01} *
                    #      [prod_{i=l+1}^{k-1} P_i^A(0)] * P_k^A(1)

                    prob = 1.0

                    # Product over i=1 to l-1
                    for i in range(1, l):
                        idx = i - 1
                        prob *= p_joint[idx, 0, 0]

                    # p_l^{01}
                    prob *= p_joint[l-1, 0, 1]

                    # Product over i=l+1 to k-1
                    for i in range(l+1, k):
                        idx = i - 1
                        prob *= P_A[idx, 0]

                    # P_k^A(1)
                    prob *= P_A[k-1, 1]

                # Add contribution (k * l)^n * Pr(F_A=k, F_B=l)
                result += (k * l)**n * prob

        return result
