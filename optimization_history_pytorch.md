1. Let's start with optimizing pytorch, dense
```
Timer unit: 1e-09 s

Total time: 5.858 s
File: /tmp/ipykernel_3295823/820451408.py
Function: random_control_2 at line 66

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    66                                               def random_control_2(self,p_ctrl,p_proj):
    67                                                   '''the competition between chaotic and random, where the projection can only be applied after the unitary
    68                                                   Notation: L-1 is the last digits'''
    69                                                   # vec=self.vec_history[-1]
    70                                                   
    71       512  214798111.0 419527.6      3.7          p= self.get_prob_tensor([self.L-1],self.vec)
    72                                           
    73       512     309404.0    604.3      0.0          pool = ["C0","C1","chaotic"]
    74       512     809182.0   1580.4      0.0          probabilities = [p_ctrl * p[(self.L-1,0)], p_ctrl * p[(self.L-1,1)],  1- p_ctrl]
    75                                           
    76       512   40593941.0  79285.0      0.7          op = self.rng.choice(pool,p=probabilities)
    77                                           
    78       512    1022814.0   1997.7      0.0          op_list= {"C0":partial(self.control_map,bL=0),
    79       512     311825.0    609.0      0.0                  "C1":partial(self.control_map,bL=1),
    80       512    1553779.0   3034.7      0.0                  f"P{self.L-1}0":partial(self.projection_map,pos=self.L-1,n=0),
    81       512     706671.0   1380.2      0.0                  f"P{self.L-1}1":partial(self.projection_map,pos=self.L-1,n=1),
    82       512     583063.0   1138.8      0.0                  f"P{self.L-2}0":partial(self.projection_map,pos=self.L-2,n=0),
    83       512     556889.0   1087.7      0.0                  f"P{self.L-2}1":partial(self.projection_map,pos=self.L-2,n=1),
    84       512     261403.0    510.6      0.0                  "chaotic":self.Bernoulli_map,
    85       512     272331.0    531.9      0.0                  "I":lambda x:x
    86                                                           }
    87       512 4487442675.0 8764536.5     76.6          self.vec=op_list[op](self.vec)
    88       512    5861309.0  11447.9      0.1          self.update_history(self.vec,op)
    89                                           
    90       273     184815.0    677.0      0.0          if op=="chaotic":
    91       546     465992.0    853.5      0.0              for pos in [self.L-1,self.L-2]:
    92       546  217571062.0 398481.8      3.7                  p_2=self.get_prob_tensor([pos], self.vec)
    93       546    1283529.0   2350.8      0.0                  pool_2=["I",f"P{pos}0",f"P{pos}1"]
    94       546     644436.0   1180.3      0.0                  probabilities_2=[1-p_proj, p_proj * p_2[(pos,0)], p_proj *  p_2[(pos,1)],]
    95       546   36183006.0  66269.2      0.6                  op_2 = self.rng.choice(pool_2,p=probabilities_2)
    96       546  844365546.0 1546457.0     14.4                  self.vec=op_list[op_2](self.vec)
    97       546    2216786.0   4060.0      0.0                  self.update_history(self.vec,op_2)
```
2. Let's see the break down
control_map: 3.17299
Bernoulli_map: 1.15578
projection_map:  0.927144
```
Timer unit: 1e-09 s

Total time: 3.17299 s
File: /tmp/ipykernel_4040775/820451408.py
Function: control_map at line 41

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    41                                               def control_map(self,vec,bL):
    42                                                   '''control map depends on the outcome of the measurement of bL'''
    43                                                   # projection on the last bits
    44       239   22583824.0  94493.0      0.7          vec=self.P_tensor(vec,bL)
    45       126      88518.0    702.5      0.0          if bL==1:
    46       126  230048998.0 1825785.7      7.3              vec=self.XL_tensor(vec)
    47       239  877155907.0 3670108.4     27.6          vec=self.normalize(vec)
    48                                                   # right shift 
    49       239   11931866.0  49924.1      0.4          vec=self.T_tensor(vec,left=False)
    50                                           
    51       239  124228702.0 519785.4      3.9          assert np.abs(vec[1]).sum() == 0, f'first qubit is not zero ({np.abs(vec[1]).sum()}) after right shift '
    52                                           
    53                                                   # Adder
    54       239  776398955.0 3248531.2     24.5          new_idx,old_idx,new_idx_complement=self.adder()
    55       239 1130437886.0 4729865.6     35.6          vec=self.adder_tensor(vec,new_idx,old_idx,new_idx_complement)
    56                                                   
    57       239     112981.0    472.7      0.0          return vec
```

```
Timer unit: 1e-09 s

Total time: 1.15578 s
File: /tmp/ipykernel_4040775/820451408.py
Function: Bernoulli_map at line 36

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    36                                               def Bernoulli_map(self,vec):
    37       273    7806342.0  28594.7      0.7          vec=self.T_tensor(vec,left=True)
    38       273 1147842984.0 4204553.1     99.3          vec=self.S_tensor(vec,rng=self.rng)
    39       273     128589.0    471.0      0.0          return vec
```

```
Timer unit: 1e-09 s

Total time: 0.927144 s
File: /tmp/ipykernel_4040775/820451408.py
Function: projection_map at line 59

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    59                                               def projection_map(self,vec,pos,n):
    60                                                   '''projection to `pos` with outcome of `n`
    61                                                   note that here is 0-index, and pos=L-1 is the last bit'''
    62       249   24882470.0  99929.6      2.7          vec=self.P_tensor(vec,n,pos)
    63       249  902149720.0 3623091.2     97.3          vec=self.normalize(vec)
    64       249     111346.0    447.2      0.0          return vec
```

3. Let focused on normalize
```
Timer unit: 1e-09 s

Total time: 0.861958 s
File: /tmp/ipykernel_4040775/820451408.py
Function: normalize at line 169

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   169                                               def normalize(self,vec):
   170                                                   # normalization after projection
   171       488  624444072.0 1279598.5     72.4          norm=np.sqrt(torch.tensordot(vec.conj(),vec,dims=(list(range(self.L_T)),list(range(self.L_T)))))
   172       488   17182661.0  35210.4      2.0          assert norm != 0 , f'Cannot normalize: norm is zero {norm}'
   173       488  220331756.0 451499.5     25.6          return vec/norm
```
