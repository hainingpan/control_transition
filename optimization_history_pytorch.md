1. Get innerprod, is very slow

`%lprun -f CT_tensor.inner_prob ct.random_control_2(.0, 0)`
```
Timer unit: 1e-09 s

Total time: 0.184821 s
File: /tmp/ipykernel_83901/959504200.py
Function: inner_prob at line 191

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   191                                               def inner_prob(self,vec,pos,n_list=[0]):
   192                                                   '''probability of `vec` of measuring 0 at L
   193                                                   convert the vector to tensor (2,2,..), take about the specific pos-th index, and flatten to calculate the inner product'''
   194         3      15028.0   5009.3      0.0          idx_list=[slice(None)]*self.L_T
   195         3      10250.0   3416.7      0.0          for p,n in zip(pos,n_list):
   196         3       2855.0    951.7      0.0              idx_list[p]=n
   197         3     492983.0 164327.7      0.3          vec_0=vec[tuple(idx_list)]
   198         3  184007888.0 61335962.7     99.6          inner_prod=torch.tensordot(vec_0.conj(),vec_0,dims=(list(range(vec_0.dim())),list(range(vec_0.dim())))).item()
   199                                           
   200         3     224223.0  74741.0      0.1          assert np.abs(inner_prod.imag)<self._eps, f'probability for outcome 0 is not real {inner_prod}'
   201         3       5370.0   1790.0      0.0          inner_prod=inner_prod.real
   202         3      11041.0   3680.3      0.0          assert inner_prod>-self._eps, f'probability for outcome 0 is not positive {inner_prod}'
   203         3      29226.0   9742.0      0.0          inner_prod=max(0,inner_prod)
   204         3      10961.0   3653.7      0.0          assert inner_prod<1+self._eps, f'probability for outcome 1 is not smaller than 1 {inner_prod}'
   205         3       9209.0   3069.7      0.0          inner_prod=min(inner_prod,1)
   206         3       2023.0    674.3      0.0          return inner_prod
```
1.1 Now I change tensordot to flatten version, nothing really changes
Timer unit: 1e-09 s

Total time: 0.175922 s
File: /tmp/ipykernel_83901/746283062.py
Function: inner_prob at line 191

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   191                                               def inner_prob(self,vec,pos,n_list):
   192                                                   '''probability of `vec` of measuring `n_list` at `pos`
   193                                                   convert the vector to tensor (2,2,..), take about the specific pos-th index, and flatten to calculate the inner product'''
   194         3      16152.0   5384.0      0.0          idx_list=[slice(None)]*self.L_T
   195         3       9448.0   3149.3      0.0          for p,n in zip(pos,n_list):
   196         3       2425.0    808.3      0.0              idx_list[p]=n
   197         3     447835.0 149278.3      0.3          vec_0=vec[tuple(idx_list)]
   198         3     519423.0 173141.0      0.3          vec_0=vec_0.contiguous().view(-1)
   199         3  174703683.0 58234561.0     99.3          inner_prod=torch.inner(vec_0,vec_0.conj()).item()
   200                                                   # inner_prod=torch.sum(vec_0*vec_0.conj()).item()
   201                                           
   202                                                   # inner_prod=torch.tensordot(vec_0.conj(),vec_0,dims=(list(range(vec_0.dim())),list(range(vec_0.dim())))).item()
   203                                           
   204         3     166101.0  55367.0      0.1          assert np.abs(inner_prod.imag)<self._eps, f'probability for outcome 0 is not real {inner_prod}'
   205         3       4979.0   1659.7      0.0          inner_prod=inner_prod.real
   206         3       7615.0   2538.3      0.0          assert inner_prod>-self._eps, f'probability for outcome 0 is not positive {inner_prod}'
   207         3      23525.0   7841.7      0.0          inner_prod=max(0,inner_prod)
   208         3      10932.0   3644.0      0.0          assert inner_prod<1+self._eps, f'probability for outcome 1 is not smaller than 1 {inner_prod}'
   209         3       8266.0   2755.3      0.0          inner_prod=min(inner_prod,1)
   210         3       1974.0    658.0      0.0          return inner_prod
1.2 change it to torch.sum, still nothing significant matters

Timer unit: 1e-09 s

Total time: 0.186279 s
File: /tmp/ipykernel_83901/3216232220.py
Function: inner_prob at line 191

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   191                                               def inner_prob(self,vec,pos,n_list):
   192                                                   '''probability of `vec` of measuring `n_list` at `pos`
   193                                                   convert the vector to tensor (2,2,..), take about the specific pos-th index, and flatten to calculate the inner product'''
   194         3      16041.0   5347.0      0.0          idx_list=[slice(None)]*self.L_T
   195         3       9789.0   3263.0      0.0          for p,n in zip(pos,n_list):
   196         3       2634.0    878.0      0.0              idx_list[p]=n
   197         3     542828.0 180942.7      0.3          vec_0=vec[tuple(idx_list)]
   198         3     930739.0 310246.3      0.5          vec_0=vec_0.contiguous().view(-1)
   199                                                   # inner_prod=torch.inner(vec_0,vec_0.conj()).item()
   200         3  184601323.0 61533774.3     99.1          inner_prod=torch.sum(vec_0*vec_0.conj()).item()
   201                                           
   202                                                   # inner_prod=torch.tensordot(vec_0.conj(),vec_0,dims=(list(range(vec_0.dim())),list(range(vec_0.dim())))).item()
   203                                           
   204         3     132687.0  44229.0      0.1          assert np.abs(inner_prod.imag)<self._eps, f'probability for outcome 0 is not real {inner_prod}'
   205         3       3907.0   1302.3      0.0          inner_prod=inner_prod.real
   206         3       5612.0   1870.7      0.0          assert inner_prod>-self._eps, f'probability for outcome 0 is not positive {inner_prod}'
   207         3      19137.0   6379.0      0.0          inner_prod=max(0,inner_prod)
   208         3       6793.0   2264.3      0.0          assert inner_prod<1+self._eps, f'probability for outcome 1 is not smaller than 1 {inner_prod}'
   209         3       6262.0   2087.3      0.0          inner_prod=min(inner_prod,1)
   210         3       1612.0    537.3      0.0          return inner_prod