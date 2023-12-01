
def monitor(func):
    def wrapper(self, *args, **kwargs):
        # Call the original function
        result = func(self, *args, **kwargs)
        # Check if monitoring is enabled
        if self.monitor:
            # Call the monitoring function with the result
            if not hasattr(self, 'EE_history'):
                # self.EE_history=[self.von_Neumann_entropy_pure([self.L],vec=result)]
                self.EE_history=[self.half_system_entanglement_entropy(vec=result)]
            else:
                # self.EE_history.append(self.von_Neumann_entropy_pure([self.L],vec=result))
                self.EE_history.append(self.half_system_entanglement_entropy(vec=result))
        # Return the original result
        return result
    return wrapper

def dec2bin(x,L):
    """convert a float number x in [0,1) to the binary form with maximal length of L, where the leading 0 as integer part is truncated. Example, 1/3 is 010101...

    Parameters
    ----------
    x : float, 0<=x<1
        float number to be converted
    L : int
        length of the binary form

    Returns
    -------
    numpy.array, shape=(L,)
        array of binary form
    """
    assert 0<=x<1, f'{x} is not in [0,1)'
    return int(x*(1<<L))

def U(n,rng=None,size=1):
    """Calculate Haar random unitary matrix of dimension `n`. The method is based on QR decomposition of a random matrix with Gaussian entries.

    Parameters
    ----------
    n : int
        dimension of the unitary matrix
    rng : numpy.random.Generator, optional
        Random generator, by default None
    size : int, optional
        Number of unitary matrix, by default 1

    Returns
    -------
    numpy.array, shape=(size,n,n)
        Haar random unitary matrix
    """
    import numpy as np
    import scipy

    if rng is None:
        rng=np.random.default_rng(None)
    return scipy.stats.unitary_group.rvs(n,random_state=rng,size=size)

def Haar_state(L,ensemble,rng=None,k=1):
    """Generate `k` orthogonal Haar random states, using the method in https://quantumcomputing.stackexchange.com/questions/15754/confusion-about-the-output-distribution-of-haar-random-quantum-states

    Parameters
    ----------
    L : int
        Length of the system, gives 2**L dimension of the Hilbert space
    ensemble : int
        Ensemble size
    rng : np.random.Generator, optional
        Random generator, by default None
    k : int, optional, {0,1}
        Number of orthrogonal Haar random state, by default 1

    Returns
    -------
    np.array, shape=(2**L,k,ensemble)
        The orthogonal `k` Haar random states
    """
    import numpy as np
    
    if rng is None:
        rng=np.random.default_rng(None)
    assert k in [1,2], f'k ({k}) is not 1 or 2'
    state=rng.normal(size=(2**L,2,k,ensemble)) # (wf, re/im, k, ensemble)
    z=state[:,0,:,:]+1j*state[:,1,:,:] # (wf, k, ensemble)
    norm=np.sqrt((np.abs(z[:,0,:])**2).sum(axis=0)) # (ensemble,)
    z[:,0,:]/=norm
    if k==2:
        z[:,1,:]-=(z[:,0,:].conj()*z[:,1,:]).sum(axis=0)*z[:,0,:]
        norm=np.sqrt((np.abs(z[:,1,:])**2).sum(axis=0))
        z[:,1,:]/=norm
    return z

from functools import lru_cache
@lru_cache(maxsize=None)
def bin_pad(x,L):
    """Convert an integer `x` to binary form with length `L`, with 0 padding to the left.

    Parameters
    ----------
    x : int
        integer in decimal form
    L : int
        length of the binary form

    Returns
    -------
    str
        bitstring of length `L` in binary form
    """
    return (bin(x)[2:]).rjust(L,'0')
