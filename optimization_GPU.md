Control_map
---

Using cuda
Timer unit: 1e-09 s

Total time: 0.0726057 s
File: /tmp/ipykernel_577049/910965828.py
Function: control_map at line 617

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   617                                               def control_map(self,vec,bL):
   618                                                   '''control map depends on the outcome of the measurement of bL'''
   619                                                   # projection on the last bits
   620         4     572908.0 143227.0      0.8          self.P_tensor_(vec,bL)
   621         2       1633.0    816.5      0.0          if bL==1:
   622         2    5684841.0 2842420.5      7.8              self.XL_tensor_(vec)
   623         4   39138144.0 9784536.0     53.9          self.normalize_(vec)
   624                                                   # right shift 
   625         4     633384.0 158346.0      0.9          vec=self.T_tensor(vec,left=False)
   626                                           
   627                                                   # Adder
   628         4    6809686.0 1702421.5      9.4          new_idx,old_idx=self.adder()
   629         4      53643.0  13410.8      0.1          if not vec.is_contiguous():
   630         4     593437.0 148359.2      0.8              vec=vec.contiguous()
   631         4   19116246.0 4779061.5     26.3          self.adder_tensor_(vec,new_idx,old_idx)
   632                                                   
   633         4       1784.0    446.0      0.0          return vec

Using cuda
Timer unit: 1e-09 s

Total time: 0.139426 s
File: /tmp/ipykernel_588651/3413241233.py
Function: run_tensor at line 1

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     1                                           def run_tensor():
     2         1   10663785.0 10663785.0      7.6      ct=CT_tensor(L=18,gpu=True,seed=list(range(100)),x0=None,ancilla=False,history=False,complex128=True)
     3                                               # ct=ct.control_map(ct.vec,1)
     4         2       3326.0   1663.0      0.0      for _ in range(2):
     5         2  128758885.0 64379442.5     92.3          ct.random_control(1,0)
     6                                               # return ct

GPU:
Using cuda
Timer unit: 1e-09 s

Total time: 1.51596 s
File: /tmp/ipykernel_644927/2083826031.py
Function: run_tensor at line 1

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     1                                           def run_tensor():
     2         1   26722931.0 26722931.0      1.8      ct=CT_tensor(L=18,gpu=True,seed=list(range(100)),x0=None,ancilla=False,history=False,complex128=True)
     3                                               # ct=ct.control_map(ct.vec,1)
     4        10      19238.0   1923.8      0.0      for _ in range(10):
     5        10 1489219065.0 148921906.5     98.2          ct.random_control(1,0)
     6                                               # return ct


## 1.1 Now profile why adder_gpu is even slower than adder_cpu::
**This is adder_cpu:**

```
Using cuda
Timer unit: 1e-09 s

Total time: 0.0135574 s
File: /tmp/ipykernel_644927/2357798511.py
Function: adder_cpu at line 326

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   326                                               def adder_cpu(self):
   327                                                   ''' This is not a full adder, which assume the leading digit in the input bitstring is zero (because of the T^{-1}R_L, the leading bit should always be zero).'''
   328         1     135862.0 135862.0      1.0          if self.xj==set([Fraction(1,3),Fraction(2,3)]):
   329         1      37703.0  37703.0      0.3              int_1_6=(int(Fraction(1,6)*2**self.L)|1)
   330         1      11021.0  11021.0      0.1              int_1_3=(int(Fraction(1,3)*2**self.L))
   331                                                           
   332                                                       
   333         1    1987368.0 1987368.0     14.7              old_idx=np.arange(2**(self.L-1)).reshape((2,-1))
   334         1      26231.0  26231.0      0.2              adder_idx=np.array([[int_1_6],[int_1_3]])
   335         1    2221861.0 2221861.0     16.4              new_idx=(old_idx+adder_idx)
   336                                                       # handle the extra attractors, if 1..0x1, then 1..0(1-x)1, if 0..1x0, then 0..1(1-x)0 [shouldn't enter this branch..]
   337         1    4343619.0 4343619.0     32.0              mask_1=(new_idx&(1<<self.L-1) == (1<<self.L-1)) & (new_idx&(1<<2) == (0)) & (new_idx&(1) == (1))
   338         1    1409442.0 1409442.0     10.4              mask_2=(new_idx&(1<<self.L-1) == (0)) & (new_idx&(1<<2) == (1<<2)) & (new_idx&(1) == (0))
   339                                           
   340         1    3382653.0 3382653.0     25.0              new_idx[mask_1+mask_2]=new_idx[mask_1+mask_2]^(0b10)
   341                                           
   342                                           
   343         1       1623.0   1623.0      0.0              return new_idx, old_idx
   344                                                   if self.xj==set([0]):
   345                                                       return np.array([]), np.array([]),
   ```

**This is adder_gpu:**

Using cuda
Timer unit: 1e-09 s

Total time: 0.000718336 s
File: /tmp/ipykernel_644927/2779305524.py
Function: adder_gpu at line 299

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   299                                               def adder_gpu(self):
   300                                                   ''' This is not a full adder, which assume the leading digit in the input bitstring is zero (because of the T^{-1}R_L, the leading bit should always be zero).'''
   301         1      61248.0  61248.0      8.5          if self.xj==set([Fraction(1,3),Fraction(2,3)]):
   302                                                       # bin_1_6=dec2bin(Fraction(1,6), self.L)
   303                                                       # bin_1_6[-1]=1
   304                                                       # bin_1_3=dec2bin(Fraction(1,3), self.L)
   305                                                       # int_1_6=int(''.join(map(str,bin_1_6)),2)
   306                                                       # int_1_3=int(''.join(map(str,bin_1_3)),2)
   307         1      19658.0  19658.0      2.7              int_1_6=(int(Fraction(1,6)*2**self.L)|1)
   308         1       9929.0   9929.0      1.4              int_1_3=(int(Fraction(1,3)*2**self.L))
   309                                                           
   310                                                       # old_idx=torch.vstack([torch.arange(2**(self.L-2)),torch.arange(2**(self.L-2),2**(self.L-1))])
   311                                                       
   312         1      77770.0  77770.0     10.8              old_idx=torch.arange(2**(self.L-1),device=self.device).view((2,-1))
   313         1      49736.0  49736.0      6.9              adder_idx=torch.tensor([[int_1_6],[int_1_3]],device=self.device)
   314         1      34025.0  34025.0      4.7              new_idx=(old_idx+adder_idx)
   315                                                       # handle the extra attractors, if 1..0x1, then 1..0(1-x)1, if 0..1x0, then 0..1(1-x)0 [shouldn't enter this branch..]
   316         1     135001.0 135001.0     18.8              mask_1=(new_idx&(1<<self.L-1) == (1<<self.L-1)) & (new_idx&(1<<2) == (0)) & (new_idx&(1) == (1))
   317         1      81397.0  81397.0     11.3              mask_2=(new_idx&(1<<self.L-1) == (0)) & (new_idx&(1<<2) == (1<<2)) & (new_idx&(1) == (0))
   318                                           
   319         1     248950.0 248950.0     34.7              new_idx[mask_1+mask_2]=new_idx[mask_1+mask_2]^(0b10)
   320                                           
   321                                           
   322         1        622.0    622.0      0.1              return new_idx, old_idx
   323                                                   if self.xj==set([0]):
   324                                                       return torch.tensor([]), torch.tensor([]),

## 1.2 This is actually very interesting then:
The adder_gpu is indeed much faster than adder_cpu, but then why does the total time of adder_gpu is longer than adder_cpu?? 
Let's look at the breakdown on control level 


**GPU**

```
Using cuda
Timer unit: 1e-09 s

Total time: 8.9933 s
File: /tmp/ipykernel_644927/2779305524.py
Function: random_control at line 93

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    93                                               def random_control(self,p_ctrl,p_proj,vec=None):
    94                                                   '''the competition between chaotic and random, where the projection can only be applied after the unitary
    95                                                   Notation: L-1 is the last digits'''
    96        50      46021.0    920.4      0.0          if vec is None:
    97        50      39858.0    797.2      0.0              vec=self.vec
    98                                           
    99        50   31584493.0 631689.9      0.4          ctrl_idx_dict=self.generate_binary(torch.arange(self.rng.shape[0]), p_ctrl)
   100        50      56677.0   1133.5      0.0          ctrl_0_idx_dict={False:[],True:[]}
   101        50      41120.0    822.4      0.0          if len(ctrl_idx_dict[True])>0:
   102        50 2257612663.0 45152253.3     25.1              p_0= self.inner_prob(vec=vec[...,ctrl_idx_dict[True]],pos=[self.L-1],n_list=[0]) # prob for 0
   103        50   72122774.0 1442455.5      0.8              ctrl_0_idx_dict=self.generate_binary(ctrl_idx_dict[True], p_0)
   104       100     262513.0   2625.1      0.0              for key,idx in ctrl_0_idx_dict.items():
   105       100     255909.0   2559.1      0.0                  if len(idx)>0:
   106       100 6630619379.0 66306193.8     73.7                      vec[...,idx]=self.op_list[f'C{0*key+1*(1-key)}'](vec[...,idx])
   107                                           
   108        50      63390.0   1267.8      0.0          proj_idx_dict={} # {pos: {True: .., False:..}} whether pos is projected
   109        50      49181.0    983.6      0.0          proj_0_idx_dict={} # {pos: {True:.., False: ..}} if projected, whether it is projected to 0 
   110        50     253363.0   5067.3      0.0          if len(ctrl_idx_dict[False])>0:
   111                                                       vec[...,ctrl_idx_dict[False]]=self.op_list['chaotic'](vec[...,ctrl_idx_dict[False]],self.rng[ctrl_idx_dict[False]])
   112                                                       for pos in [self.L-1,self.L-2]:
   113                                                           proj_idx_dict[pos]=self.generate_binary(ctrl_idx_dict[False], p_proj)
   114                                                           if len(proj_idx_dict[pos][True])>0:
   115                                                               p_2 = self.inner_prob(vec=vec[...,proj_idx_dict[pos][True]],pos=[pos], n_list=[0])
   116                                                               proj_0_idx_dict[pos]=self.generate_binary(proj_idx_dict[pos][True], p_2)
   117                                                               for key,idx in proj_0_idx_dict[pos].items():
   118                                                                   if len(idx)>0:
   119                                                                       vec[...,idx]=self.op_list[f'P{pos}{0*key+1*(1-key)}'](vec[...,idx])
   120        50     294644.0   5892.9      0.0          self.update_history(vec,ctrl_idx_dict,ctrl_0_idx_dict,proj_idx_dict,proj_0_idx_dict)
   ```

   **CPU**
   ```
   Using cuda
Timer unit: 1e-09 s

Total time: 9.13219 s
File: /tmp/ipykernel_644927/2357798511.py
Function: random_control at line 93

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    93                                               def random_control(self,p_ctrl,p_proj,vec=None):
    94                                                   '''the competition between chaotic and random, where the projection can only be applied after the unitary
    95                                                   Notation: L-1 is the last digits'''
    96        50      46501.0    930.0      0.0          if vec is None:
    97        50      41585.0    831.7      0.0              vec=self.vec
    98                                           
    99        50   33599782.0 671995.6      0.4          ctrl_idx_dict=self.generate_binary(torch.arange(self.rng.shape[0]), p_ctrl)
   100        50      57513.0   1150.3      0.0          ctrl_0_idx_dict={False:[],True:[]}
   101        50      51061.0   1021.2      0.0          if len(ctrl_idx_dict[True])>0:
   102        50 2252890134.0 45057802.7     24.7              p_0= self.inner_prob(vec=vec[...,ctrl_idx_dict[True]],pos=[self.L-1],n_list=[0]) # prob for 0
   103        50   79480558.0 1589611.2      0.9              ctrl_0_idx_dict=self.generate_binary(ctrl_idx_dict[True], p_0)
   104       100     245680.0   2456.8      0.0              for key,idx in ctrl_0_idx_dict.items():
   105       100     228383.0   2283.8      0.0                  if len(idx)>0:
   106       100 6764996486.0 67649964.9     74.1                      vec[...,idx]=self.op_list[f'C{0*key+1*(1-key)}'](vec[...,idx])
   107                                           
   108        50      47601.0    952.0      0.0          proj_idx_dict={} # {pos: {True: .., False:..}} whether pos is projected
   109        50      30260.0    605.2      0.0          proj_0_idx_dict={} # {pos: {True:.., False: ..}} if projected, whether it is projected to 0 
   110        50     204372.0   4087.4      0.0          if len(ctrl_idx_dict[False])>0:
   111                                                       vec[...,ctrl_idx_dict[False]]=self.op_list['chaotic'](vec[...,ctrl_idx_dict[False]],self.rng[ctrl_idx_dict[False]])
   112                                                       for pos in [self.L-1,self.L-2]:
   113                                                           proj_idx_dict[pos]=self.generate_binary(ctrl_idx_dict[False], p_proj)
   114                                                           if len(proj_idx_dict[pos][True])>0:
   115                                                               p_2 = self.inner_prob(vec=vec[...,proj_idx_dict[pos][True]],pos=[pos], n_list=[0])
   116                                                               proj_0_idx_dict[pos]=self.generate_binary(proj_idx_dict[pos][True], p_2)
   117                                                               for key,idx in proj_0_idx_dict[pos].items():
   118                                                                   if len(idx)>0:
   119                                                                       vec[...,idx]=self.op_list[f'P{pos}{0*key+1*(1-key)}'](vec[...,idx])
   120        50     266898.0   5338.0      0.0          self.update_history(vec,ctrl_idx_dict,ctrl_0_idx_dict,proj_idx_dict,proj_0_idx_dict)
   ```

   1.3 Ok, it seems, there is not much difference in using either way.

   For GPU,


   ```
   Using cuda
Timer unit: 1e-09 s

Total time: 2.11838 s
File: /tmp/ipykernel_644927/2779305524.py
Function: inner_prob at line 204

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   204                                               def inner_prob(self,vec,pos,n_list):
   205                                                   '''probability of `vec` of measuring `n_list` at `pos`
   206                                                   convert the vector to tensor (2,2,..), take about the specific pos-th index, and flatten to calculate the inner product'''
   207        50    1177486.0  23549.7      0.1          idx_list=np.array([slice(None)]*self.L_T)
   208                                                   # for p,n in zip(pos,n_list):
   209                                                   #     idx_list[p]=n
   210        50   15506987.0 310139.7      0.7          idx_list[pos]=n_list
   211        50    4112838.0  82256.8      0.2          vec_0=vec[tuple(idx_list)]
   212        50   35051446.0 701028.9      1.7          inner_prod=torch.einsum(vec_0.conj(),[...,0],vec_0,[...,0],[0]) # overhead??
   213                                           
   214        50 2044147105.0 40882942.1     96.5          assert torch.all(torch.abs(inner_prod.imag)<self._eps), f'probability for outcome 0 is not real {inner_prod}'
   215        50    2566369.0  51327.4      0.1          inner_prod=inner_prod.real
   216                                                   # assert torch.all(inner_prod>-self._eps), f'probability for outcome 0 is not positive {inner_prod}'
   217                                                   # inner_prod=torch.maximum(0,inner_prod)
   218                                                   # assert torch.all(inner_prod<1+self._eps), f'probability for outcome 1 is not smaller than 1 {inner_prod}'
   219                                                   # inner_prod=torch.minimum(inner_prod,1)
   220        50   15792854.0 315857.1      0.7          inner_prod=torch.clamp_(inner_prod,min=0,max=1)
   221        50      27514.0    550.3      0.0          return inner_prod
   ```

   Ok assert takes a lot of time, I can turn off it, which gives:
   ```
   Using cuda
Timer unit: 1e-09 s

Total time: 0.0216606 s
File: /tmp/ipykernel_644927/1159384219.py
Function: inner_prob at line 204

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   204                                               def inner_prob(self,vec,pos,n_list):
   205                                                   '''probability of `vec` of measuring `n_list` at `pos`
   206                                                   convert the vector to tensor (2,2,..), take about the specific pos-th index, and flatten to calculate the inner product'''
   207        50    1013949.0  20279.0      4.7          idx_list=np.array([slice(None)]*self.L_T)
   208                                                   # for p,n in zip(pos,n_list):
   209                                                   #     idx_list[p]=n
   210        50    1273649.0  25473.0      5.9          idx_list[pos]=n_list
   211        50    3753693.0  75073.9     17.3          vec_0=vec[tuple(idx_list)]
   212        50   13040881.0 260817.6     60.2          inner_prod=torch.einsum(vec_0.conj(),[...,0],vec_0,[...,0],[0]) # overhead??
   213                                           
   214                                                   # assert torch.all(torch.abs(inner_prod.imag)<self._eps), f'probability for outcome 0 is not real {inner_prod}'
   215        50     814584.0  16291.7      3.8          inner_prod=inner_prod.real
   216                                                   # assert torch.all(inner_prod>-self._eps), f'probability for outcome 0 is not positive {inner_prod}'
   217                                                   # inner_prod=torch.maximum(0,inner_prod)
   218                                                   # assert torch.all(inner_prod<1+self._eps), f'probability for outcome 1 is not smaller than 1 {inner_prod}'
   219                                                   # inner_prod=torch.minimum(inner_prod,1)
   220        50    1751596.0  35031.9      8.1          inner_prod=torch.clamp_(inner_prod,min=0,max=1)
   221        50      12223.0    244.5      0.1          return inner_prod
   
   ```

   ```
   Using cuda
Timer unit: 1e-09 s

Total time: 8.96759 s
File: /tmp/ipykernel_644927/1159384219.py
Function: random_control at line 93

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    93                                               def random_control(self,p_ctrl,p_proj,vec=None):
    94                                                   '''the competition between chaotic and random, where the projection can only be applied after the unitary
    95                                                   Notation: L-1 is the last digits'''
    96        50      42534.0    850.7      0.0          if vec is None:
    97        50      37925.0    758.5      0.0              vec=self.vec
    98                                           
    99        50   36002962.0 720059.2      0.4          ctrl_idx_dict=self.generate_binary(torch.arange(self.rng.shape[0]), p_ctrl)
   100        50      58771.0   1175.4      0.0          ctrl_0_idx_dict={False:[],True:[]}
   101        50      41500.0    830.0      0.0          if len(ctrl_idx_dict[True])>0:
   102        50  236821720.0 4736434.4      2.6              p_0= self.inner_prob(vec=vec[...,ctrl_idx_dict[True]],pos=[self.L-1],n_list=[0]) # prob for 0
   103        50 2074524221.0 41490484.4     23.1              ctrl_0_idx_dict=self.generate_binary(ctrl_idx_dict[True], p_0)
   104       100     276310.0   2763.1      0.0              for key,idx in ctrl_0_idx_dict.items():
   105       100     263892.0   2638.9      0.0                  if len(idx)>0:
   106       100 6618941440.0 66189414.4     73.8                      vec[...,idx]=self.op_list[f'C{0*key+1*(1-key)}'](vec[...,idx])
   107                                           
   108        50      44815.0    896.3      0.0          proj_idx_dict={} # {pos: {True: .., False:..}} whether pos is projected
   109        50      32024.0    640.5      0.0          proj_0_idx_dict={} # {pos: {True:.., False: ..}} if projected, whether it is projected to 0 
   110        50     206318.0   4126.4      0.0          if len(ctrl_idx_dict[False])>0:
   111                                                       vec[...,ctrl_idx_dict[False]]=self.op_list['chaotic'](vec[...,ctrl_idx_dict[False]],self.rng[ctrl_idx_dict[False]])
   112                                                       for pos in [self.L-1,self.L-2]:
   113                                                           proj_idx_dict[pos]=self.generate_binary(ctrl_idx_dict[False], p_proj)
   114                                                           if len(proj_idx_dict[pos][True])>0:
   115                                                               p_2 = self.inner_prob(vec=vec[...,proj_idx_dict[pos][True]],pos=[pos], n_list=[0])
   116                                                               proj_0_idx_dict[pos]=self.generate_binary(proj_idx_dict[pos][True], p_2)
   117                                                               for key,idx in proj_0_idx_dict[pos].items():
   118                                                                   if len(idx)>0:
   119                                                                       vec[...,idx]=self.op_list[f'P{pos}{0*key+1*(1-key)}'](vec[...,idx])
   120        50     295363.0   5907.3      0.0          self.update_history(vec,ctrl_idx_dict,ctrl_0_idx_dict,proj_idx_dict,proj_0_idx_dict)
   ```

```
Using cuda
Timer unit: 1e-09 s

Total time: 2.10355 s
File: /tmp/ipykernel_644927/1159384219.py
Function: generate_binary at line 364

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   364                                               def generate_binary(self,idx_list,p):
   365                                                   '''Generate boolean list, given probability `p` and seed `self.rng[idx]`'''
   366        50     205526.0   4110.5      0.0          if isinstance(p, float) or isinstance(p, int):
   367        50    3337408.0  66748.2      0.2              p_list=torch.tensor([p]*len(idx_list))
   368                                                   else:
   369        50     706741.0  14134.8      0.0              assert len(idx_list) == len(p), f'len of idx_list {len(idx_list)} is not same as len of p {len(p)}'
   370        50      13763.0    275.3      0.0              p_list=p
   371                                           
   372       100     155122.0   1551.2      0.0          idx_dict={True:[],False:[]}
   373     10000   22696462.0   2269.6      1.1          for idx,p in zip(idx_list,p_list):
   374     10000 2076394585.0 207639.5     98.7              idx_dict[self.rng[idx].random()<=p.item()].append(idx) # Here p should be a scalar but it returns a tensor
   375       100      36572.0    365.7      0.0          return idx_dict
   ```

   ```
   Using cuda
Timer unit: 1e-09 s

Total time: 2.14646 s
File: /tmp/ipykernel_644927/814821096.py
Function: generate_binary at line 364

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   364                                               def generate_binary(self,idx_list,p):
   365                                                   '''Generate boolean list, given probability `p` and seed `self.rng[idx]`'''
   366        50     173485.0   3469.7      0.0          if isinstance(p, float) or isinstance(p, int):
   367        50    3307455.0  66149.1      0.2              p_list=torch.tensor([p]*len(idx_list))
   368                                                   else:
   369        50     590686.0  11813.7      0.0              assert len(idx_list) == len(p), f'len of idx_list {len(idx_list)} is not same as len of p {len(p)}'
   370        50      11783.0    235.7      0.0              p_list=p
   371                                           
   372       100     135644.0   1356.4      0.0          idx_dict={True:[],False:[]}
   373     10000   21970873.0   2197.1      1.0          for idx,p in zip(idx_list,p_list):
   374     10000   29188599.0   2918.9      1.4              random=self.rng[idx].random()
   375     10000 2085067259.0 208506.7     97.1              p=p.item()
   376     10000    5976210.0    597.6      0.3              idx_dict[random<=p].append(idx) # Here p should be a scalar but it returns a tensor
   377       100      33798.0    338.0      0.0          return idx_dict
   ```

   ```
   Using cuda
Timer unit: 1e-09 s

Total time: 2.05713 s
File: /tmp/ipykernel_644927/1838672530.py
Function: generate_binary at line 364

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   364                                               def generate_binary(self,idx_list,p):
   365                                                   '''Generate boolean list, given probability `p` and seed `self.rng[idx]`'''
   366        50     170402.0   3408.0      0.0          if isinstance(p, float) or isinstance(p, int):
   367        50    1110945.0  22218.9      0.1              p_list=([p]*len(idx_list))
   368                                                   else:
   369        50     811416.0  16228.3      0.0              assert len(idx_list) == len(p), f'len of idx_list {len(idx_list)} is not same as len of p {len(p)}'
   370        50 2002222700.0 40044454.0     97.3              p_list=p.cpu().numpy()
   371                                           
   372       100     360988.0   3609.9      0.0          idx_dict={True:[],False:[]}
   373     10000   10231282.0   1023.1      0.5          for idx,p in zip(idx_list,p_list):
   374     10000   27500589.0   2750.1      1.3              random=self.rng[idx].random()
   375     10000   14699459.0   1469.9      0.7              idx_dict[random<=p].append(idx) # Here p should be a scalar but it returns a tensor
   376       100      25720.0    257.2      0.0          return idx_dict
   ```
   Still very slow..., especially copying is bottleneck

   ```
   Using cuda
Timer unit: 1e-09 s

Total time: 1.8229 s
File: /tmp/ipykernel_658456/285314670.py
Function: generate_binary at line 364

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   364                                               def generate_binary(self,idx_list,p):
   365                                                   '''Generate boolean list, given probability `p` and seed `self.rng[idx]`
   366                                                   scalar `p` is verbose, but this is for consideration of speed'''
   367       100     340065.0   3400.7      0.0          idx_dict={True:[],False:[]}
   368        50     203159.0   4063.2      0.0          if isinstance(p, float) or isinstance(p, int):
   369     10000   17563576.0   1756.4      1.0              for idx in idx_list:
   370     10000   34084178.0   3408.4      1.9                  random=self.rng[idx].random()
   371     10000    3430811.0    343.1      0.2                  boolean=(random<=p)
   372     10000    5461916.0    546.2      0.3                  idx_dict[boolean].append(idx) 
   373                                                   else:
   374        50    1297760.0  25955.2      0.1              assert len(idx_list) == len(p), f'len of idx_list {len(idx_list)} is not same as len of p {len(p)}'
   375     10000   20280643.0   2028.1      1.1              for idx,p in zip(idx_list,p):
   376     10000   73964022.0   7396.4      4.1                  random=self.rng[idx].random()
   377     10000 1650603496.0 165060.3     90.5                  boolean=torch.equal((random<=p),self.tensor_true)
   378     10000   15624832.0   1562.5      0.9                  idx_dict[boolean].append(idx) 
   379       100      48364.0    483.6      0.0          return idx_dict
   ```

   Not particular sure why, but this does shorten the running time
   ```
   Using cuda
Timer unit: 1e-09 s

Total time: 5.48488 s
File: /tmp/ipykernel_658456/285314670.py
Function: random_control at line 94

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    94                                               def random_control(self,p_ctrl,p_proj,vec=None):
    95                                                   '''the competition between chaotic and random, where the projection can only be applied after the unitary
    96                                                   Notation: L-1 is the last digits'''
    97        50      47075.0    941.5      0.0          if vec is None:
    98        50      47703.0    954.1      0.0              vec=self.vec
    99                                           
   100        50   37715921.0 754318.4      0.7          ctrl_idx_dict=self.generate_binary(torch.arange(self.rng.shape[0]), p_ctrl)
   101        50      59725.0   1194.5      0.0          if len(ctrl_idx_dict[True])>0:
   102        50  106022310.0 2120446.2      1.9              p_0= self.inner_prob(vec=vec[...,ctrl_idx_dict[True]],pos=[self.L-1],n_list=[0]) # prob for 0
   103        50 1487073587.0 29741471.7     27.1              ctrl_0_idx_dict=self.generate_binary(ctrl_idx_dict[True], p_0)
   104       100     345413.0   3454.1      0.0              for key,idx in ctrl_0_idx_dict.items():
   105       100     284233.0   2842.3      0.0                  if len(idx)>0:
   106       100 3852620186.0 38526201.9     70.2                      vec[...,idx]=self.op_list[f'C{0*key+1*(1-key)}'](vec[...,idx])
   107                                           
   108        50      44432.0    888.6      0.0          proj_idx_dict={} # {pos: {True: .., False:..}} whether pos is projected
   109        50      17629.0    352.6      0.0          proj_0_idx_dict={} # {pos: {True:.., False: ..}} if projected, whether it is projected to 0 
   110        50     259844.0   5196.9      0.0          if len(ctrl_idx_dict[False])>0:
   111                                                       vec[...,ctrl_idx_dict[False]]=self.op_list['chaotic'](vec[...,ctrl_idx_dict[False]],self.rng[ctrl_idx_dict[False]])
   112                                                       for pos in [self.L-1,self.L-2]:
   113                                                           proj_idx_dict[pos]=self.generate_binary(ctrl_idx_dict[False], p_proj)
   114                                                           if len(proj_idx_dict[pos][True])>0:
   115                                                               p_2 = self.inner_prob(vec=vec[...,proj_idx_dict[pos][True]],pos=[pos], n_list=[0])
   116                                                               proj_0_idx_dict[pos]=self.generate_binary(proj_idx_dict[pos][True], p_2)
   117                                                               for key,idx in proj_0_idx_dict[pos].items():
   118                                                                   if len(idx)>0:
   119                                                                       vec[...,idx]=self.op_list[f'P{pos}{0*key+1*(1-key)}'](vec[...,idx])
   120        50     339332.0   6786.6      0.0          self.update_history(vec,ctrl_idx_dict,ctrl_0_idx_dict,proj_idx_dict,proj_0_idx_dict)
   ```

It's the best I can do so far, let see `control_map`

```
Using cuda
Timer unit: 1e-09 s

Total time: 3.57962 s
File: /tmp/ipykernel_658456/3575020279.py
Function: control_map at line 69

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    69                                               def control_map(self,vec,bL):
    70                                                   '''control map depends on the outcome of the measurement of bL'''
    71                                                   # projection on the last bits
    72       100    8566648.0  85666.5      0.2          self.P_tensor_(vec,bL)
    73        50      42176.0    843.5      0.0          if bL==1:
    74        50  245454462.0 4909089.2      6.9              self.XL_tensor_(vec)
    75       100 1873180176.0 18731801.8     52.3          self.normalize_(vec)
    76                                                   # right shift 
    77       100   10428316.0 104283.2      0.3          vec=self.T_tensor(vec,left=False)
    78                                           
    79                                                   # Adder
    80                                                   
    81       100     327081.0   3270.8      0.0          if not vec.is_contiguous():
    82       100    9911910.0  99119.1      0.3              vec=vec.contiguous()
    83       100 1431644161.0 14316441.6     40.0          self.adder_tensor_(vec,self.new_idx,self.old_idx)
    84                                                   
    85       100      64458.0    644.6      0.0          return vec
```

Ok, one by one : first is `normalize_`

```
Using cuda
Timer unit: 1e-09 s

Total time: 1.86836 s
File: /tmp/ipykernel_658456/3575020279.py
Function: normalize_ at line 197

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   197                                               def normalize_(self,vec):
   198                                                   '''normalization after projection'''
   199                                                   # norm=torch.sqrt(torch.tensordot(vec.conj(),vec,dims=(list(range(self.L_T)),list(range(self.L_T)))))
   200       100   31222906.0 312229.1      1.7          norm=torch.sqrt(torch.einsum(vec.conj(),[...,0],vec,[...,0],[0]))
   201                                           
   202       100 1831178718.0 18311787.2     98.0          assert torch.all(norm != 0) , f'Cannot normalize: norm is zero {norm}'
   203       100    5959181.0  59591.8      0.3          vec/=norm
```

Didn't expect assert take that much of time. suppress it now. Later I would want it to have a version with a knob to turn on and off all assertion the same time.

```
Using cuda
Timer unit: 1e-09 s

Total time: 0.0481718 s
File: /tmp/ipykernel_658456/1154358494.py
Function: normalize_ at line 197

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   197                                               def normalize_(self,vec):
   198                                                   '''normalization after projection'''
   199                                                   # norm=torch.sqrt(torch.tensordot(vec.conj(),vec,dims=(list(range(self.L_T)),list(range(self.L_T)))))
   200       100   34915940.0 349159.4     72.5          norm=torch.sqrt(torch.einsum(vec.conj(),[...,0],vec,[...,0],[0]))
   201                                           
   202                                                   # assert torch.all(norm != 0) , f'Cannot normalize: norm is zero {norm}'
   203       100   13255825.0 132558.2     27.5          vec/=norm
   ```

   Now the total time is saved, go back to `control_map` again.

   ```
   Using cuda
Timer unit: 1e-09 s

Total time: 3.68878 s
File: /tmp/ipykernel_658456/1154358494.py
Function: control_map at line 69

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    69                                               def control_map(self,vec,bL):
    70                                                   '''control map depends on the outcome of the measurement of bL'''
    71                                                   # projection on the last bits
    72       100   10233722.0 102337.2      0.3          self.P_tensor_(vec,bL)
    73        50      44027.0    880.5      0.0          if bL==1:
    74        50  241326066.0 4826521.3      6.5              self.XL_tensor_(vec)
    75       100   49472281.0 494722.8      1.3          self.normalize_(vec)
    76                                                   # right shift 
    77       100    6956794.0  69567.9      0.2          vec=self.T_tensor(vec,left=False)
    78                                           
    79                                                   # Adder
    80                                                   
    81       100     277509.0   2775.1      0.0          if not vec.is_contiguous():
    82       100    3406023.0  34060.2      0.1              vec=vec.contiguous()
    83       100 3376958478.0 33769584.8     91.5          self.adder_tensor_(vec,self.new_idx,self.old_idx)
    84                                                   
    85       100     105553.0   1055.5      0.0          return vec
   ```
Before going to the large proportion `adder_tensor`, I want to see what happened for `XL_tensor_`

```
Using cuda
Timer unit: 1e-09 s

Total time: 0.248047 s
File: /tmp/ipykernel_658456/1154358494.py
Function: XL_tensor_ at line 224

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   224                                               def XL_tensor_(self,vec):
   225                                                   '''directly swap 0 and 1'''
   226        50      92304.0   1846.1      0.0          if not self.ancilla:
   227                                                       # vec=vec[...,[1,0]]
   228        50  247955175.0 4959103.5    100.0              vec[...,[0,1],:]=vec[...,[1,0],:]
   229                                           
   230                                                   else:
   231                                                       # vec=vec[...,[1,0],:]
   232                                                       vec[...,[0,1],:,:]=vec[...,[1,0],:,:]
   233                                                   # return vec
```

Well, one choice is to use torch roll

```
Using cuda
Timer unit: 1e-09 s

Total time: 0.00173471 s
File: /tmp/ipykernel_658456/3294710110.py
Function: XL_tensor_ at line 224

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   224                                               def XL_tensor_(self,vec):
   225                                                   '''directly swap 0 and 1'''
   226        50      75410.0   1508.2      4.3          if not self.ancilla:
   227                                                       # vec[...,[0,1],:]=vec[...,[1,0],:]
   228        50    1659304.0  33186.1     95.7              vec=torch.roll(vec,1,self.L)
   229                                           
   230                                                   else:
   231                                                       # vec[...,[0,1],:,:]=vec[...,[1,0],:,:]
   232                                                       vec=torch.roll(vec,1,self.L)
   233                                           
   234                                                   # return vec
```

Ok, it is pretty fast, to my surprise. 
Back to `control_map`

```
Using cuda
Timer unit: 1e-09 s

Total time: 3.45037 s
File: /tmp/ipykernel_658456/1115732402.py
Function: control_map at line 69

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    69                                               def control_map(self,vec,bL):
    70                                                   '''control map depends on the outcome of the measurement of bL'''
    71                                                   # projection on the last bits
    72        59    7714544.0 130755.0      0.2          self.P_tensor_(vec,bL)
    73        50      43485.0    869.7      0.0          if bL==1:
    74        50    1933532.0  38670.6      0.1              vec=self.XL_tensor_(vec)
    75        59   35988493.0 609974.5      1.0          self.normalize_(vec)
    76                                                   # right shift 
    77        59    3510696.0  59503.3      0.1          vec=self.T_tensor(vec,left=False)
    78                                           
    79                                                   # Adder
    80                                                   
    81        59     152356.0   2582.3      0.0          if not vec.is_contiguous():
    82        59    1979482.0  33550.5      0.1              vec=vec.contiguous()
    83        59 3399009309.0 57610327.3     98.5          self.adder_tensor_(vec,self.new_idx,self.old_idx)
    84                                                   
    85        59      43002.0    728.8      0.0          return vec
```

Ok, not too bad, the last target `adder_tensor_`

```
Using cuda
Timer unit: 1e-09 s

Total time: 3.07507 s
File: /tmp/ipykernel_658456/1465800595.py
Function: adder_tensor_ at line 351

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   351                                               def adder_tensor_(self,vec,new_idx,old_idx):
   352        59     624007.0  10576.4      0.0          new_idx=new_idx.flatten()
   353        59     253945.0   4304.2      0.0          old_idx=old_idx.flatten()
   354        59     239356.0   4056.9      0.0          if (new_idx).shape[0]>0 and (old_idx).shape[0]>0:
   355                                                       
   356        59     588769.0   9979.1      0.0              vec_flatten=vec.view((-1,vec.shape[-1]))    # create a reference
   357        59      23099.0    391.5      0.0              if self.ancilla:
   358                                                           new_idx=torch.hstack((new_idx<<1,(new_idx<<1)+1))
   359                                                           old_idx=torch.hstack((old_idx<<1,(old_idx<<1)+1))
   360                                           
   361        59    3618091.0  61323.6      0.1              vec_flatten[new_idx,:]=vec_flatten[old_idx,:]
   362                                                       
   363        59    2047017.0  34695.2      0.1              not_new_map=torch.ones(2**(self.L_T),dtype=bool,device=self.device)
   364        59 3061775153.0 51894494.1     99.6              not_new_map[new_idx]=False
   365        59    5896733.0  99944.6      0.2              vec_flatten[not_new_map,:]=0
```

Ok, cleraly Line 364 is very slow, to my surprise.. Maybe I should just figure out a way that I can directly obtain the index of `not_new_map`?

At least it does not have evaluate each time.

```
Using cuda
Timer unit: 1e-09 s

Total time: 0.0118857 s
File: /tmp/ipykernel_658456/839176996.py
Function: adder_tensor_ at line 350

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   350                                               def adder_tensor_(self,vec):
   351       100     956355.0   9563.5      8.0          new_idx=self.new_idx.flatten()
   352       100     396689.0   3966.9      3.3          old_idx=self.old_idx.flatten()
   353       100     138875.0   1388.8      1.2          not_new_idx=self.not_new_idx.flatten()
   354       100     345408.0   3454.1      2.9          if (new_idx).shape[0]>0 and (old_idx).shape[0]>0:
   355                                                       
   356       100     756684.0   7566.8      6.4              vec_flatten=vec.view((-1,vec.shape[-1]))    
   357       100      40562.0    405.6      0.3              if self.ancilla:
   358                                                           new_idx=torch.hstack((new_idx<<1,(new_idx<<1)+1))
   359                                                           old_idx=torch.hstack((old_idx<<1,(old_idx<<1)+1))
   360                                           
   361       100    5718130.0  57181.3     48.1              vec_flatten[new_idx,:]=vec_flatten[old_idx,:]
   362                                                       
   363       100    3532982.0  35329.8     29.7              vec_flatten[not_new_idx,:]=0
   ```

   May be `torch.roll` can help again here. **but maybe later**

   Now go back to `control_map`:

   ```
   Using cuda
Timer unit: 1e-09 s

Total time: 0.0530875 s
File: /tmp/ipykernel_658456/839176996.py
Function: control_map at line 69

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    69                                               def control_map(self,vec,bL):
    70                                                   '''control map depends on the outcome of the measurement of bL'''
    71                                                   # projection on the last bits
    72       100    5943767.0  59437.7     11.2          self.P_tensor_(vec,bL)
    73        50      41567.0    831.3      0.1          if bL==1:
    74        50    1434078.0  28681.6      2.7              vec=self.XL_tensor_(vec)
    75       100   26712201.0 267122.0     50.3          self.normalize_(vec)
    76                                                   # right shift 
    77       100    4276126.0  42761.3      8.1          vec=self.T_tensor(vec,left=False)
    78                                           
    79                                                   # Adder
    80                                                   
    81       100     188444.0   1884.4      0.4          if not vec.is_contiguous():
    82       100    2354138.0  23541.4      4.4              vec=vec.contiguous()
    83       100   12107215.0 121072.1     22.8          self.adder_tensor_(vec)
    84                                                   
    85       100      29992.0    299.9      0.1          return vec
   ```
3.45037->0.0530875. This is good

Finally, just to make sure, let's see what `P_tensor_` says
```
Using cuda
Timer unit: 1e-09 s

Total time: 0.00629689 s
File: /tmp/ipykernel_658456/839176996.py
Function: P_tensor_ at line 237

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   237                                               def P_tensor_(self,vec,n,pos=None):
   238                                                   '''directly set zero at tensor[...,0] =0 for n==1 and tensor[...,1] =0 for n==0'
   239                                                   This is an in-placed operation
   240                                                   '''
   241                                                   # vec_tensor=vec.reshape((2,)*self.L_T)
   242       100     153100.0   1531.0      2.4          if pos is None or pos==self.L-1:
   243                                                       # project the last site
   244       100      81745.0    817.5      1.3              if not self.ancilla:
   245       100    5906056.0  59060.6     93.8                  vec[...,1-n,:]=0
   246                                                       else:
   247                                                           vec[...,1-n,:,:]=0
   248       100     155992.0   1559.9      2.5          if pos == self.L-2:
   249                                                       if not self.ancilla:
   250                                                           vec[...,1-n,:,:]=0
   251                                                       else:
   252                                                           vec[...,1-n,:,:,:]=0
   253                                                   # return vec
```

Seems reasonable, not any good idea of what to do at current stage.

Go back to `random_control`:

```
Using cuda
Timer unit: 1e-09 s

Total time: 0.715313 s
File: /tmp/ipykernel_658456/839176996.py
Function: random_control at line 94

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    94                                               def random_control(self,p_ctrl,p_proj,vec=None):
    95                                                   '''the competition between chaotic and random, where the projection can only be applied after the unitary
    96                                                   Notation: L-1 is the last digits'''
    97        50      43407.0    868.1      0.0          if vec is None:
    98        50      47585.0    951.7      0.0              vec=self.vec
    99                                           
   100        50   42042641.0 840852.8      5.9          ctrl_idx_dict=self.generate_binary(torch.arange(self.rng.shape[0]), p_ctrl)
   101        50      23077.0    461.5      0.0          ctrl_0_idx_dict={}
   102        50      47693.0    953.9      0.0          if len(ctrl_idx_dict[True])>0:
   103        50   29130710.0 582614.2      4.1              p_0= self.inner_prob(vec=vec[...,ctrl_idx_dict[True]],pos=[self.L-1],n_list=[0]) # prob for 0
   104        50  538609346.0 10772186.9     75.3              ctrl_0_idx_dict=self.generate_binary(ctrl_idx_dict[True], p_0)
   105       100     316438.0   3164.4      0.0              for key,idx in ctrl_0_idx_dict.items():
   106       100     258068.0   2580.7      0.0                  if len(idx)>0:
   107       100  104409199.0 1044092.0     14.6                      vec[...,idx]=self.op_list[f'C{0*key+1*(1-key)}'](vec[...,idx])
   108                                           
   109        50      20632.0    412.6      0.0          proj_idx_dict={} # {pos: {True: .., False:..}} whether pos is projected
   110        50      17304.0    346.1      0.0          proj_0_idx_dict={} # {pos: {True:.., False: ..}} if projected, whether it is projected to 0 
   111        50     138611.0   2772.2      0.0          if len(ctrl_idx_dict[False])>0:
   112                                                       vec[...,ctrl_idx_dict[False]]=self.op_list['chaotic'](vec[...,ctrl_idx_dict[False]],self.rng[ctrl_idx_dict[False]])
   113                                                       for pos in [self.L-1,self.L-2]:
   114                                                           proj_idx_dict[pos]=self.generate_binary(ctrl_idx_dict[False], p_proj)
   115                                                           if len(proj_idx_dict[pos][True])>0:
   116                                                               p_2 = self.inner_prob(vec=vec[...,proj_idx_dict[pos][True]],pos=[pos], n_list=[0])
   117                                                               proj_0_idx_dict[pos]=self.generate_binary(proj_idx_dict[pos][True], p_2)
   118                                                               for key,idx in proj_0_idx_dict[pos].items():
   119                                                                   if len(idx)>0:
   120                                                                       vec[...,idx]=self.op_list[f'P{pos}{0*key+1*(1-key)}'](vec[...,idx])
   121        50     207985.0   4159.7      0.0          self.update_history(vec,ctrl_idx_dict,ctrl_0_idx_dict,proj_idx_dict,proj_0_idx_dict)
```

Coming back to `generate_binary`, well it is costly but I don't know whether there is a better way?? **Maybe consider it later.**


Ok, this profiling process is a bit funny... I restart it, and now different result shows:

```
Using cuda
Timer unit: 1e-09 s

Total time: 5.08226 s
File: /tmp/ipykernel_665259/1334432072.py
Function: random_control at line 94

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    94                                               def random_control(self,p_ctrl,p_proj,vec=None):
    95                                                   '''the competition between chaotic and random, where the projection can only be applied after the unitary
    96                                                   Notation: L-1 is the last digits'''
    97        50      57937.0   1158.7      0.0          if vec is None:
    98        50      58052.0   1161.0      0.0              vec=self.vec
    99                                           
   100        50   45104702.0 902094.0      0.9          ctrl_idx_dict=self.generate_binary(torch.arange(self.rng.shape[0]), p_ctrl)
   101        50      16415.0    328.3      0.0          ctrl_0_idx_dict={}
   102        50      62030.0   1240.6      0.0          if len(ctrl_idx_dict[True])>0:
   103        50  191167323.0 3823346.5      3.8              p_0= self.inner_prob(vec=vec[...,ctrl_idx_dict[True]],pos=[self.L-1],n_list=[0]) # prob for 0
   104        50 1501362297.0 30027245.9     29.5              ctrl_0_idx_dict=self.generate_binary(ctrl_idx_dict[True], p_0)
   105       100     306442.0   3064.4      0.0              for key,idx in ctrl_0_idx_dict.items():
   106        59     126271.0   2140.2      0.0                  if len(idx)>0:
   107        59 3343035311.0 56661615.4     65.8                      vec[...,idx]=self.op_list[f'C{0*key+1*(1-key)}'](vec[...,idx])
   108                                           
   109        50      78913.0   1578.3      0.0          proj_idx_dict={} # {pos: {True: .., False:..}} whether pos is projected
   110        50      40059.0    801.2      0.0          proj_0_idx_dict={} # {pos: {True:.., False: ..}} if projected, whether it is projected to 0 
   111        50     440997.0   8819.9      0.0          if len(ctrl_idx_dict[False])>0:
   112                                                       vec_tmp=vec[...,ctrl_idx_dict[False]]
   113                                                       vec_tmp=self.op_list['chaotic'](vec_tmp,self.rng[ctrl_idx_dict[False]])
   114                                                       for pos in [self.L-1,self.L-2]:
   115                                                           proj_idx_dict[pos]=self.generate_binary(ctrl_idx_dict[False], p_proj)
   116                                                           if len(proj_idx_dict[pos][True])>0:
   117                                                               p_2 = self.inner_prob(vec=vec[...,proj_idx_dict[pos][True]],pos=[pos], n_list=[0])
   118                                                               proj_0_idx_dict[pos]=self.generate_binary(proj_idx_dict[pos][True], p_2)
   119                                                               for key,idx in proj_0_idx_dict[pos].items():
   120                                                                   if len(idx)>0:
   121                                                                       vec[...,idx]=self.op_list[f'P{pos}{0*key+1*(1-key)}'](vec[...,idx])
   122        50     399597.0   7991.9      0.0          self.update_history(vec,ctrl_idx_dict,ctrl_0_idx_dict,proj_idx_dict,proj_0_idx_dict)

```

Here, I think I need to split the slicing and operations.

```
Using cuda
Timer unit: 1e-09 s

Total time: 4.67045 s
File: /tmp/ipykernel_665259/3465118112.py
Function: random_control at line 94

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    94                                               def random_control(self,p_ctrl,p_proj,vec=None):
    95                                                   '''the competition between chaotic and random, where the projection can only be applied after the unitary
    96                                                   Notation: L-1 is the last digits'''
    97        50      72398.0   1448.0      0.0          if vec is None:
    98        50      51731.0   1034.6      0.0              vec=self.vec
    99                                           
   100        50   46593384.0 931867.7      1.0          ctrl_idx_dict=self.generate_binary(torch.arange(self.rng.shape[0]), p_ctrl)
   101        50      19893.0    397.9      0.0          ctrl_0_idx_dict={}
   102        50      64522.0   1290.4      0.0          if len(ctrl_idx_dict[True])>0:
   103        50 1675913329.0 33518266.6     35.9              vec_ctrl=vec[...,ctrl_idx_dict[True]]
   104        50   35864169.0 717283.4      0.8              p_0= self.inner_prob(vec=vec_ctrl,pos=[self.L-1],n_list=[0]) # prob for 0
   105        50 1537799947.0 30755998.9     32.9              ctrl_0_idx_dict=self.generate_binary(ctrl_idx_dict[True], p_0)
   106       100     331197.0   3312.0      0.0              for key,idx in ctrl_0_idx_dict.items():
   107       100     297338.0   2973.4      0.0                  if len(idx)>0:
   108       100 1279910963.0 12799109.6     27.4                      vec_ctrl_i=vec[...,idx]
   109       100   93029633.0 930296.3      2.0                      vec_ctrl_i=self.op_list[f'C{0*key+1*(1-key)}'](vec_ctrl_i)
   110                                           
   111        50      15262.0    305.2      0.0          proj_idx_dict={} # {pos: {True: .., False:..}} whether pos is projected
   112        50      15151.0    303.0      0.0          proj_0_idx_dict={} # {pos: {True:.., False: ..}} if projected, whether it is projected to 0 
   113        50     195952.0   3919.0      0.0          if len(ctrl_idx_dict[False])>0:
   114                                                       vec_chaotic=vec[...,ctrl_idx_dict[False]]
   115                                                       vec_chaotic=self.op_list['chaotic'](vec_chaotic,self.rng[ctrl_idx_dict[False]])
   116                                                       for pos in [self.L-1,self.L-2]:
   117                                                           proj_idx_dict[pos]=self.generate_binary(ctrl_idx_dict[False], p_proj)
   118                                                           if len(proj_idx_dict[pos][True])>0:
   119                                                               vec_p=vec[...,proj_idx_dict[pos][True]]
   120                                                               p_2 = self.inner_prob(vec=vec_p,pos=[pos], n_list=[0])
   121                                                               proj_0_idx_dict[pos]=self.generate_binary(proj_idx_dict[pos][True], p_2)
   122                                                               for key,idx in proj_0_idx_dict[pos].items():
   123                                                                   if len(idx)>0:
   124                                                                       vec_proj=vec[...,idx]
   125                                                                       vec_proj=self.op_list[f'P{pos}{0*key+1*(1-key)}'](vec_proj)
   126        50     274811.0   5496.2      0.0          self.update_history(vec,ctrl_idx_dict,ctrl_0_idx_dict,proj_idx_dict,proj_0_idx_dict)

```
Now I split them, it is a bit faster, I guess because they do not need to modfiy the metadata during slice. Well, one possibility is the way `idx` are store, right now, they are store like `[tensor()...]`, this may not be optimal because `tensor([..])` seems to be better, let try this. So in order to do this, I just change the way how `dict` is generated:

from 
```
Using cuda
Timer unit: 1e-09 s

Total time: 1.54922 s
File: /tmp/ipykernel_665259/2437828240.py
Function: generate_binary at line 370

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   370                                               def generate_binary(self,idx_list,p):
   371                                                   '''Generate boolean list, given probability `p` and seed `self.rng[idx]`
   372                                                   scalar `p` is verbose, but this is for consideration of speed'''
   373       100     267047.0   2670.5      0.0          idx_dict={True:[],False:[]}
   374        50     160708.0   3214.2      0.0          if isinstance(p, float) or isinstance(p, int):
   375     10000   12650827.0   1265.1      0.8              for idx in idx_list:
   376     10000   24937664.0   2493.8      1.6                  random=self.rng[idx].random()
   377     10000    2744247.0    274.4      0.2                  boolean=(random<=p)
   378     10000    4267711.0    426.8      0.3                  idx_dict[boolean].append(idx) 
   379                                                   else:
   380        50    1077273.0  21545.5      0.1              assert len(idx_list) == len(p), f'len of idx_list {len(idx_list)} is not same as len of p {len(p)}'
   381     10000   18474431.0   1847.4      1.2              for idx,p in zip(idx_list,p):
   382     10000   48757329.0   4875.7      3.1                  random=self.rng[idx].random()
   383     10000 1425646516.0 142564.7     92.0                  boolean=torch.equal((random<=p),self.tensor_true)
   384     10000   10194450.0   1019.4      0.7                  idx_dict[boolean].append(idx)
   385                                                   
   386       100      39844.0    398.4      0.0          return idx_dict
```

to 
```
Using cuda
Timer unit: 1e-09 s

Total time: 4.36925 s
File: /tmp/ipykernel_665259/446057583.py
Function: generate_binary at line 370

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   370                                               def generate_binary(self,idx_list,p):
   371                                                   '''Generate boolean list, given probability `p` and seed `self.rng[idx]`
   372                                                   scalar `p` is verbose, but this is for consideration of speed'''
   373                                                   # idx_dict={True:[],False:[]}
   374       100     153737.0   1537.4      0.0          true_list=[]
   375       100      34514.0    345.1      0.0          false_list=[]
   376        50     163098.0   3262.0      0.0          if isinstance(p, float) or isinstance(p, int):
   377     10000   13830863.0   1383.1      0.3              for idx in idx_list:
   378     10000   26296388.0   2629.6      0.6                  random=self.rng[idx].random()
   379                                                           # boolean=(random<=p)
   380                                                           # idx_dict[boolean].append(idx)
   381     10000    2822149.0    282.2      0.1                  if random<=p:
   382     10000    3668227.0    366.8      0.1                      true_list.append(idx)
   383                                                           else:
   384                                                               false_list.append(idx)
   385                                                   else:
   386        50     645799.0  12916.0      0.0              assert len(idx_list) == len(p), f'len of idx_list {len(idx_list)} is not same as len of p {len(p)}'
   387     10000   29744355.0   2974.4      0.7              for idx,p in zip(idx_list,p):
   388     10000 1095362994.0 109536.3     25.1                  random=self.rng[idx].random()
   389                                                           # boolean=torch.equal((random<=p),self.tensor_true)
   390                                                           # idx_dict[boolean].append(idx)
   391      5600  146959928.0  26242.8      3.4                  if random<=p:
   392      4400    2730217.0    620.5      0.1                      true_list.append(idx)
   393                                                           else:
   394      5600    5757637.0   1028.1      0.1                      false_list.append(idx)
   395                                           
   396       100 3040974077.0 30409740.8     69.6          idx_dict={True:torch.tensor(true_list,dtype=int,device=self.device),False:torch.tensor(false_list,dtype=int,device=self.device)}
   397       100     103632.0   1036.3      0.0          return idx_dict
```

Ok, this does not mean anything useful, because it makes generating much slower, though the slicing is superfast. Another method, which might be useful is that, completely drop the np.rng and use GPU random seed, to generate all random in gpu. 

```Using cuda
Timer unit: 1e-09 s

Total time: 4.5884 s
File: /tmp/ipykernel_665259/446057583.py
Function: random_control at line 94

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    94                                               def random_control(self,p_ctrl,p_proj,vec=None):
    95                                                   '''the competition between chaotic and random, where the projection can only be applied after the unitary
    96                                                   Notation: L-1 is the last digits'''
    97        50      53574.0   1071.5      0.0          if vec is None:
    98        50      26142.0    522.8      0.0              vec=self.vec
    99                                           
   100        50 3011589541.0 60231790.8     65.6          ctrl_idx_dict=self.generate_binary(torch.arange(self.rng.shape[0]), p_ctrl)
   101        50      68555.0   1371.1      0.0          ctrl_0_idx_dict={}
   102        50    1336967.0  26739.3      0.0          if len(ctrl_idx_dict[True])>0:
   103        50    7030889.0 140617.8      0.2              vec_ctrl=vec[...,ctrl_idx_dict[True]]
   104        50   28443258.0 568865.2      0.6              p_0= self.inner_prob(vec=vec_ctrl,pos=[self.L-1],n_list=[0]) # prob for 0
   105        50 1477471198.0 29549424.0     32.2              ctrl_0_idx_dict=self.generate_binary(ctrl_idx_dict[True], p_0)
   106       100     223443.0   2234.4      0.0              for key,idx in ctrl_0_idx_dict.items():
   107       100    1510476.0  15104.8      0.0                  if len(idx)>0:
   108       100    6142604.0  61426.0      0.1                      vec_ctrl_i=vec[...,idx]
   109       100   53917141.0 539171.4      1.2                      vec_ctrl_i=self.op_list[f'C{0*key+1*(1-key)}'](vec_ctrl_i)
   110                                           
   111        50      12751.0    255.0      0.0          proj_idx_dict={} # {pos: {True: .., False:..}} whether pos is projected
   112        50      10141.0    202.8      0.0          proj_0_idx_dict={} # {pos: {True:.., False: ..}} if projected, whether it is projected to 0 
   113        50     376610.0   7532.2      0.0          if len(ctrl_idx_dict[False])>0:
   114                                                       vec_chaotic=vec[...,ctrl_idx_dict[False]]
   115                                                       vec_chaotic=self.op_list['chaotic'](vec_chaotic,self.rng[ctrl_idx_dict[False]])
   116                                                       for pos in [self.L-1,self.L-2]:
   117                                                           proj_idx_dict[pos]=self.generate_binary(ctrl_idx_dict[False], p_proj)
   118                                                           if len(proj_idx_dict[pos][True])>0:
   119                                                               vec_p=vec[...,proj_idx_dict[pos][True]]
   120                                                               p_2 = self.inner_prob(vec=vec_p,pos=[pos], n_list=[0])
   121                                                               proj_0_idx_dict[pos]=self.generate_binary(proj_idx_dict[pos][True], p_2)
   122                                                               for key,idx in proj_0_idx_dict[pos].items():
   123                                                                   if len(idx)>0:
   124                                                                       vec_proj=vec[...,idx]
   125                                                                       vec_proj=self.op_list[f'P{pos}{0*key+1*(1-key)}'](vec_proj)
   126        50     189030.0   3780.6      0.0          self.update_history(vec,ctrl_idx_dict,ctrl_0_idx_dict,proj_idx_dict,proj_0_idx_dict)
   
   ```
Alright, at this stage, I don't think I can have any method other than put the whole random generator to torch.




**Switch to Bernoulli map**

```
Using cuda
Timer unit: 1e-09 s

Total time: 7.01337 s
File: /tmp/ipykernel_658456/839176996.py
Function: random_control at line 94

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    94                                               def random_control(self,p_ctrl,p_proj,vec=None):
    95                                                   '''the competition between chaotic and random, where the projection can only be applied after the unitary
    96                                                   Notation: L-1 is the last digits'''
    97        50      21458.0    429.2      0.0          if vec is None:
    98        50      16808.0    336.2      0.0              vec=self.vec
    99                                           
   100        50   28978399.0 579568.0      0.4          ctrl_idx_dict=self.generate_binary(torch.arange(self.rng.shape[0]), p_ctrl)
   101        50      16098.0    322.0      0.0          ctrl_0_idx_dict={}
   102        50      33585.0    671.7      0.0          if len(ctrl_idx_dict[True])>0:
   103                                                       p_0= self.inner_prob(vec=vec[...,ctrl_idx_dict[True]],pos=[self.L-1],n_list=[0]) # prob for 0
   104                                                       ctrl_0_idx_dict=self.generate_binary(ctrl_idx_dict[True], p_0)
   105                                                       for key,idx in ctrl_0_idx_dict.items():
   106                                                           if len(idx)>0:
   107                                                               vec[...,idx]=self.op_list[f'C{0*key+1*(1-key)}'](vec[...,idx])
   108                                           
   109        50      11359.0    227.2      0.0          proj_idx_dict={} # {pos: {True: .., False:..}} whether pos is projected
   110        50       9167.0    183.3      0.0          proj_0_idx_dict={} # {pos: {True:.., False: ..}} if projected, whether it is projected to 0 
   111        50      21307.0    426.1      0.0          if len(ctrl_idx_dict[False])>0:
   112        50 6944098873.0 138881977.5     99.0              vec[...,ctrl_idx_dict[False]]=self.op_list['chaotic'](vec[...,ctrl_idx_dict[False]],self.rng[ctrl_idx_dict[False]])
   113       100     103624.0   1036.2      0.0              for pos in [self.L-1,self.L-2]:
   114       100   39898399.0 398984.0      0.6                  proj_idx_dict[pos]=self.generate_binary(ctrl_idx_dict[False], p_proj)
   115       100      94172.0    941.7      0.0                  if len(proj_idx_dict[pos][True])>0:
   116                                                               p_2 = self.inner_prob(vec=vec[...,proj_idx_dict[pos][True]],pos=[pos], n_list=[0])
   117                                                               proj_0_idx_dict[pos]=self.generate_binary(proj_idx_dict[pos][True], p_2)
   118                                                               for key,idx in proj_0_idx_dict[pos].items():
   119                                                                   if len(idx)>0:
   120                                                                       vec[...,idx]=self.op_list[f'P{pos}{0*key+1*(1-key)}'](vec[...,idx])
   121        50      67728.0   1354.6      0.0          self.update_history(vec,ctrl_idx_dict,ctrl_0_idx_dict,proj_idx_dict,proj_0_idx_dict)
```

Clearly, op_list is very slow, so what happened here? op_list['chaotic] very slow?? First of all, we decompose slicing and function calls:


```
Using cuda
Timer unit: 1e-09 s

Total time: 6.11356 s
File: /tmp/ipykernel_658456/1334432072.py
Function: random_control at line 94

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    94                                               def random_control(self,p_ctrl,p_proj,vec=None):
    95                                                   '''the competition between chaotic and random, where the projection can only be applied after the unitary
    96                                                   Notation: L-1 is the last digits'''
    97        50      46641.0    932.8      0.0          if vec is None:
    98        50      34938.0    698.8      0.0              vec=self.vec
    99                                           
   100        50   33721109.0 674422.2      0.6          ctrl_idx_dict=self.generate_binary(torch.arange(self.rng.shape[0]), p_ctrl)
   101        50      19185.0    383.7      0.0          ctrl_0_idx_dict={}
   102        50      44071.0    881.4      0.0          if len(ctrl_idx_dict[True])>0:
   103                                                       p_0= self.inner_prob(vec=vec[...,ctrl_idx_dict[True]],pos=[self.L-1],n_list=[0]) # prob for 0
   104                                                       ctrl_0_idx_dict=self.generate_binary(ctrl_idx_dict[True], p_0)
   105                                                       for key,idx in ctrl_0_idx_dict.items():
   106                                                           if len(idx)>0:
   107                                                               vec[...,idx]=self.op_list[f'C{0*key+1*(1-key)}'](vec[...,idx])
   108                                           
   109        50      10781.0    215.6      0.0          proj_idx_dict={} # {pos: {True: .., False:..}} whether pos is projected
   110        50      10613.0    212.3      0.0          proj_0_idx_dict={} # {pos: {True:.., False: ..}} if projected, whether it is projected to 0 
   111        50      22613.0    452.3      0.0          if len(ctrl_idx_dict[False])>0:
   112        50 5148685602.0 102973712.0     84.2              vec_tmp=vec[...,ctrl_idx_dict[False]]
   113        50  880804360.0 17616087.2     14.4              vec_tmp=self.op_list['chaotic'](vec_tmp,self.rng[ctrl_idx_dict[False]])
   114       100     134688.0   1346.9      0.0              for pos in [self.L-1,self.L-2]:
   115       100   49716023.0 497160.2      0.8                  proj_idx_dict[pos]=self.generate_binary(ctrl_idx_dict[False], p_proj)
   116       100     118082.0   1180.8      0.0                  if len(proj_idx_dict[pos][True])>0:
   117                                                               p_2 = self.inner_prob(vec=vec[...,proj_idx_dict[pos][True]],pos=[pos], n_list=[0])
   118                                                               proj_0_idx_dict[pos]=self.generate_binary(proj_idx_dict[pos][True], p_2)
   119                                                               for key,idx in proj_0_idx_dict[pos].items():
   120                                                                   if len(idx)>0:
   121                                                                       vec[...,idx]=self.op_list[f'P{pos}{0*key+1*(1-key)}'](vec[...,idx])
   122        50     188090.0   3761.8      0.0          self.update_history(vec,ctrl_idx_dict,ctrl_0_idx_dict,proj_idx_dict,proj_0_idx_dict)
```
Look slicing is definitely an issue here. So why?  Is that because vec is not contiguous at that point? That simply does not make sense... because control is the same slicing but took much little time than this. Also, 





