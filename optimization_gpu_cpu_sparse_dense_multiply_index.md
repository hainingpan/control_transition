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
- [x]. in place vs return a new [as much in-place as I can]
4. cpu vs gpu [GPU definitely at larger matrix size]


# Summary of benchmark: evo=10 steps
CPU: AMD Ryzen Threadripper PRO 5995WX 64-Cores
GPU: RTX A4000  (6144 CUDA cores, 192 Tensor cores, 48 RT cores, 16GB GDDR6 RAM)
## xj={1/3,2/3}
### ancilla = False

**Bernoulli map (p_ctrl=0, p_proj=0)**

| L:   | 26   | 28 (GPU:complex64) |
|------|------|------|
| CPU: | 32s  | 1m45s|
| GPU: | 1.6s |  2.4s|

**control map (p_ctrl=1, p_proj=0)**

| L:   | 26   |
|------|------|
| CPU: | 42s  |
| GPU: | 3.3s |

**(p_ctrl=.5, p_proj=0.)**

| L:   | 26   |
|------|------|
| CPU: | 38.5s  |
| GPU: | 3s |

## ancilla = True, no initialize

**Bernoulli map (matrix multiplication) (p_ctrl=0, p_proj=0)**

| L:   | 26    |
|------|-------|
| CPU: | 1m21s |
| GPU: | 4.6s  |

**control map (p_ctrl=1, p_proj=0)**

| L:   | 26    |
|------|-------|
| CPU: | 2m4s  |
| GPU: | 11s   |

## xj={0}
### ancilla = False

**Bernoulli map (p_ctrl=0, p_proj=0)**

| L:   | 26  |
|------|-----|
| CPU: | 27.7s   |
| GPU: | 1.6s   |

**control map (p_ctrl=1, p_proj=0)**

| L:   | 26  |
|------|-----|
| CPU: | 39s   |
| GPU: | 0.4s   |

**(p_ctrl=.5, p_proj=0)**

| L:   | 26 |
|------|----|
| CPU: | 39.2s  |
| GPU: |  1s |

### ancilla = True

**Bernoulli map (p_ctrl=0, p_proj=0)**

| L:   | 26    |
|------|-------|
| CPU: | 1m22s |
| GPU: | 5s    |


**Bernoulli map (p_ctrl=0, p_proj=0)**
| L:   | 26 |
|------|----|
| CPU: | 2m13s  |
| GPU: | 0.9s  |

**(p_ctrl=.5, p_proj=0)**

| L:   | 26 |
|------|----|
| CPU: | 1m59s  |
| GPU: | 3.5s  |


# Summary of benchmark: evo=10 steps, with multi-ensemble (1000)

**(p_ctrl=0, p_proj=0)**
| L:   | 18   |  (GPU:complex64) |
|------|------|------|
| CPU: | 1m18s  | s|
| GPU: | s |  s|

**(p_ctrl=1, p_proj=0)**
| L:   | 18   |  (GPU:complex64) |
|------|------|------|
| CPU: | 2m12s  | s|
| GPU: | s |  s|

**(p_ctrl=.5, p_proj=0)**
| L:   | 18   |  (GPU:complex64) |
|------|------|------|
| CPU(1,128):| 2m  | s|
| GPU: | 7.7s |  s|

1. profile time
2. profile GPU RAM
