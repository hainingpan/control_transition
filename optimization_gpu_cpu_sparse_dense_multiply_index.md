- [x]. which is faster, matrix multiplication or index scrambling? [tensordot does not work for sparse, even if I convert to dense, ]
    Result for L=13:
    CPU permute: 5.3 mus
    GPU permute: 5.26 mus
    CPU multiplication: 20.6ms
    GPU multiplication: 829 mus
    Thus, permute << multipliation, where gpu is generally faster than cpu.

    For L=20, 
    CPU permute: 8.51 mus
    GPU permute: 5.17 mus
    For L=31,
    CPU permute: 8 mus
    GPU permute: 8 mus

    tensor product, L=28:
    CPU : 646ms
    GPU : 500 mus

    Problem is I need to have to way to prevent it from out of GPU RAM




- [x] sparse vs dense, [sparse is not a good idea because the sparsity is actually very low in volume law, most likely to be a dense matrix]
3. in place vs return a new
4. cpu vs gpu