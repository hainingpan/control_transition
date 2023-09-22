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

 2. This time is the profiling of with the support of multi-ensemble

profile random_control for p_ctrl=0,p_proj=0
   ```
  Using cuda
Timer unit: 1e-09 s

Total time: 17.2971 s
File: /tmp/ipykernel_368513/3402788794.py
Function: random_control at line 92

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    92                                               def random_control(self,p_ctrl,p_proj,vec=None):
    93                                                   '''the competition between chaotic and random, where the projection can only be applied after the unitary
    94                                                   Notation: L-1 is the last digits'''
    95        50      70667.0   1413.3      0.0          if vec is None:
    96        50      42902.0    858.0      0.0              vec=self.vec
    97                                           
    98        50   71115219.0 1422304.4      0.4          ctrl_idx_dict=self.generate_binary(np.arange(self.rng.shape[0]), p_ctrl)
    99        50      54982.0   1099.6      0.0          ctrl_0_idx_dict={False:[],True:[]}
   100        50      54878.0   1097.6      0.0          if len(ctrl_idx_dict[True])>0:
   101                                                       p_0= self.inner_prob(vec=vec[...,ctrl_idx_dict[True]],pos=[self.L-1],n_list=[0]) # prob for 0
   102                                                       ctrl_0_idx_dict=self.generate_binary(ctrl_idx_dict[True], p_0)
   103                                                       for key,idx in ctrl_0_idx_dict.items():
   104                                                           if len(idx)>0:
   105                                                               vec[...,idx]=self.op_list[f'C{0*key+1*(1-key)}'](vec[...,idx])
   106                                           
   107        50      25686.0    513.7      0.0          proj_idx_dict={} # {pos: {True: .., False:..}} whether pos is projected
   108        50       8841.0    176.8      0.0          proj_0_idx_dict={} # {pos: {True:.., False: ..}} if projected, whether it is projected to 0 
   109        50      32323.0    646.5      0.0          if len(ctrl_idx_dict[False])>0:
   110        50      13521.0    270.4      0.0              idx_tmp=ctrl_idx_dict[False]
   111        50 13021873255.0 260437465.1     75.3              vec_tmp=vec[...,idx_tmp]
   112        50 4045363733.0 80907274.7     23.4              vec_tmp=self.op_list['chaotic'](vec_tmp,self.rng[ctrl_idx_dict[False]])
   113       100     174111.0   1741.1      0.0              for pos in [self.L-1,self.L-2]:
   114       100  157936279.0 1579362.8      0.9                  proj_idx_dict[pos]=self.generate_binary(ctrl_idx_dict[False], p_proj)
   115       100     131427.0   1314.3      0.0                  if len(proj_idx_dict[pos][True])>0:
   116                                                               p_2 = self.inner_prob(vec=vec[...,proj_idx_dict[pos][True]],pos=[pos], n_list=[0])
   117                                                               proj_0_idx_dict[pos]=self.generate_binary(proj_idx_dict[pos][True], p_2)
   118                                                               for key,idx in proj_0_idx_dict[pos].items():
   119                                                                   if len(idx)>0:
   120                                                                       vec[...,idx]=self.op_list[f'P{pos}{0*key+1*(1-key)}'](vec[...,idx])
   121        50     240831.0   4816.6      0.0          self.update_history(vec,ctrl_idx_dict,ctrl_0_idx_dict,proj_idx_dict,proj_0_idx_dict)
   ```

   2.1 It seems slicing is very slow.


   3. profiling control_map
   ```
   Using cuda
Timer unit: 1e-09 s

Total time: 11.5508 s
File: /tmp/ipykernel_368513/404598224.py
Function: random_control at line 92

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    92                                               def random_control(self,p_ctrl,p_proj,vec=None):
    93                                                   '''the competition between chaotic and random, where the projection can only be applied after the unitary
    94                                                   Notation: L-1 is the last digits'''
    95        50      54173.0   1083.5      0.0          if vec is None:
    96        50      43133.0    862.7      0.0              vec=self.vec
    97                                           
    98        50   82681806.0 1653636.1      0.7          ctrl_idx_dict=self.generate_binary(np.arange(self.rng.shape[0]), p_ctrl)
    99        50      61360.0   1227.2      0.0          ctrl_0_idx_dict={False:[],True:[]}
   100        50      52526.0   1050.5      0.0          if len(ctrl_idx_dict[True])>0:
   101        50 2948543703.0 58970874.1     25.5              p_0= self.inner_prob(vec=vec[...,ctrl_idx_dict[True]],pos=[self.L-1],n_list=[0]) # prob for 0
   102        50  174776734.0 3495534.7      1.5              ctrl_0_idx_dict=self.generate_binary(ctrl_idx_dict[True], p_0)
   103       100     227574.0   2275.7      0.0              for key,idx in ctrl_0_idx_dict.items():
   104       100     268357.0   2683.6      0.0                  if len(idx)>0:
   105       100  293783307.0 2937833.1      2.5                      vec_tmp=vec[...,idx]
   106       100 8049663509.0 80496635.1     69.7                      vec_tmp=self.op_list[f'C{0*key+1*(1-key)}'](vec_tmp)
   107                                           
   108        50      30129.0    602.6      0.0          proj_idx_dict={} # {pos: {True: .., False:..}} whether pos is projected
   109        50      40066.0    801.3      0.0          proj_0_idx_dict={} # {pos: {True:.., False: ..}} if projected, whether it is projected to 0 
   110        50     274834.0   5496.7      0.0          if len(ctrl_idx_dict[False])>0:
   111                                                       vec_tmp=vec[...,ctrl_idx_dict[False]]
   112                                                       vec_tmp=self.op_list['chaotic'](vec_tmp,self.rng[ctrl_idx_dict[False]])
   113                                                       for pos in [self.L-1,self.L-2]:
   114                                                           proj_idx_dict[pos]=self.generate_binary(ctrl_idx_dict[False], p_proj)
   115                                                           if len(proj_idx_dict[pos][True])>0:
   116                                                               p_2 = self.inner_prob(vec=vec[...,proj_idx_dict[pos][True]],pos=[pos], n_list=[0])
   117                                                               proj_0_idx_dict[pos]=self.generate_binary(proj_idx_dict[pos][True], p_2)
   118                                                               for key,idx in proj_0_idx_dict[pos].items():
   119                                                                   if len(idx)>0:
   120                                                                       vec[...,idx]=self.op_list[f'P{pos}{0*key+1*(1-key)}'](vec[...,idx])
   121        50     325038.0   6500.8      0.0          self.update_history(vec,ctrl_idx_dict,ctrl_0_idx_dict,proj_idx_dict,proj_0_idx_dict)
   ```

   Question: 1. Why 100 times at Line 106
   2. why slicing here is fast?
   3. control seems only different in XL, 





   ```
   Using cuda
Timer unit: 1e-09 s

Total time: 0.987599 s
File: /tmp/ipykernel_366037/214933894.py
Function: Bernoulli_map at line 62

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    62                                               def Bernoulli_map(self,vec,rng):
    63        10     948551.0  94855.1      0.1          vec=self.T_tensor(vec,left=True)
    64        10  986642152.0 98664215.2     99.9          vec=self.S_tensor(vec,rng=rng)
    65        10       8628.0    862.8      0.0          return vec
   ```

2.2 Go to S_tensor
```
   Using cuda
Timer unit: 1e-09 s

Total time: 0.680999 s
File: /tmp/ipykernel_366037/214933894.py
Function: S_tensor at line 267

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   267                                               def S_tensor(self,vec,rng):
   268                                                   '''Scrambler only applies to the last two indices'''
   269        10  673331663.0 67333166.3     98.9          U_4=torch.from_numpy(np.array([U(4,rng).astype(self.dtype['numpy']).reshape((2,)*4) for rng in rng]))
   270        10      20688.0   2068.8      0.0          if self.gpu:
   271        10    2842912.0 284291.2      0.4              U_4=U_4.cuda()
   272        10      13005.0   1300.5      0.0          if not self.ancilla:
   273                                                       # vec=torch.tensordot(vec,U_4,dims=([self.L-2,self.L-1],[2,3])).permute(list(range(self.L-2))+[self.L-1,self.L]+[self.L-2])
   274                                                       vec=torch.einsum(vec,[...,0,1,2],U_4,[2,3,4,0,1],[...,3,4,2])
   275                                                       return vec
   276                                                   else:
   277                                                       # vec=torch.tensordot(vec,U_4,dims=([self.L-2,self.L-1],[2,3])).permute(list(range(self.L-2))+[self.L,self.L+1]+[self.L-2,self.L-1])
   278        10    4787704.0 478770.4      0.7              vec=torch.einsum(vec,[...,0,1,2,3],U_4,[3,4,5,0,1],[...,4,5,2,3])
   279        10       3488.0    348.8      0.0              return vec
   ```


