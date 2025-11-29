import numpy as np

class CT_quantum_stat_mech_O1:
    """Optimized O(1) memory version using First Domain Wall (FDW) representation.

    State is always [1,1,...,1,0,0,...,0], so we only track FDW:
    - FDW = number of zeros from right = L - (number of 1s)
    - FDW = 0: all 1s [1,1,1,1]
    - FDW = L: all 0s [0,0,0,0]
    """
    def __init__(self, L, seed=None, seed_vec=None, seed_C=None, x0=None):
        self.L = L
        self.rng = np.random.default_rng(seed)
        self.rng_C = np.random.default_rng(seed_C) if seed_C is not None else self.rng
        self.x0 = x0
        self.FDW = self._initialize_FDW()

    def _initialize_FDW(self):
        '''"0" is for maximal mixed state as <Z>=0
        "1" is for fixed state as <Z>=1
        Initialize to all 1s → FDW = 0'''
        return 0

    def Bernouli(self):
        """Bernoulli map: shifts left and sets last 2 positions to 0.
        - FDW ≤ 1 → FDW = 2
        - FDW ≥ 2 → FDW increases by 1 (capped at L)
        """
        if self.FDW <= 1:
            self.FDW = 2
        else:
            self.FDW = min(self.FDW + 1, self.L)

    def control(self):
        """Control operation: adds a 1 on the left.
        FDW decreases by 1 (minimum 0)
        """
        self.FDW = max(self.FDW - 1, 0)

    def variance(self):
        """Variance = 1/L - <sum(vec²)>/L²
        Number of 1s = L - FDW, so sum(vec²) = L - FDW (each 1 contributes 1², each 0 contributes 0²)
        """
        num_ones = self.L - self.FDW
        return 1/self.L - num_ones / self.L**2

    def variance_FDW(self):
        """Variance of First Domain Wall position.
        For pattern [1,1,...,1,0,0,...,0] with FDW zeros:
        - q-th zero from left is at position (FDW-q+1) from right
        - Contribution: 0.5^q × (FDW-q+1)
        Returns: <k²> - <k>²

        Uses closed-form formula (simplified to avoid overflow):
        2.0 - 4^(-k) - 2^(-k) × (1 + 2k)
        """
        if self.FDW == 0:
            return 0.0

        k = self.FDW
        return 2.0 - (4.0 ** (-k)) - (2.0 ** (-k)) * (1 + 2*k)

    def random_circuit(self, p):
        """Apply random circuit: Bernoulli with prob (1-p), control with prob p"""
        prob = self.rng_C.random()
        if prob >= p:
            self.Bernouli()
        else:
            self.control()
