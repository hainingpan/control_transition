1. See in large matrix, which is the bottleneck? Is it the matrix multiplication or index shuffling?
```
Timer unit: 1e-09 s

Total time: 23.8815 s
File: /tmp/ipykernel_3549544/2655625021.py
Function: random_control_2 at line 197

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   197                                               def random_control_2(self,p_ctrl,p_proj):
   198                                                   '''the competition between chaotic and random, where the projection can only be applied after the unitary
   199                                                   Notation: L-1 is the last digits'''
   200       800 1022582044.0 1278227.6      4.3          vec=self.vec_history[-1].copy()
   201                                                   
   202       800 1955683781.0 2444604.7      8.2          p= self.get_prob_tensor([self.L-1],vec)
   203                                           
   204       800     669679.0    837.1      0.0          pool = ["C0","C1","chaotic"]
   205       800    1952495.0   2440.6      0.0          probabilities = [p_ctrl * p[(self.L-1,0)], p_ctrl * p[(self.L-1,1)],  1- p_ctrl]
   206                                           
   207       800   50154289.0  62692.9      0.2          op = self.rng.choice(pool,p=probabilities)
   208                                           
   209       800    1279865.0   1599.8      0.0          op_list= {"C0":partial(self.control_map,bL=0),
   210       800     311794.0    389.7      0.0                    "C1":partial(self.control_map,bL=1),
   211       800    2044286.0   2555.4      0.0                    f"P{self.L-1}0":partial(self.projection_map,pos=self.L-1,n=0),
   212       800     806260.0   1007.8      0.0                    f"P{self.L-1}1":partial(self.projection_map,pos=self.L-1,n=1),
   213       800     774536.0    968.2      0.0                    f"P{self.L-2}0":partial(self.projection_map,pos=self.L-2,n=0),
   214       800     756959.0    946.2      0.0                    f"P{self.L-2}1":partial(self.projection_map,pos=self.L-2,n=1),
   215       800     347086.0    433.9      0.0                    "chaotic":self.Bernoulli_map,
   216       800     454982.0    568.7      0.0                    "I":lambda x:x
   217                                                             }
   218       800 14204570129.0 17755712.7     59.5          vec=op_list[op](vec)
   219       800  123339777.0 154174.7      0.5          self.update_history(vec,op)
   220                                           
   221       423     264365.0    625.0      0.0          if op=="chaotic":
   222       846     664945.0    786.0      0.0              for pos in [self.L-1,self.L-2]:
   223       846 3845207040.0 4545162.0     16.1                  p_2=self.get_prob_tensor([pos], vec)
   224       846    1861205.0   2200.0      0.0                  pool_2=["I",f"P{pos}0",f"P{pos}1"]
   225       846    1019211.0   1204.7      0.0                  probabilities_2=[1-p_proj, p_proj * p_2[(pos,0)], p_proj *  p_2[(pos,1)],]
   226       846   44776147.0  52926.9      0.2                  op_2 = self.rng.choice(pool_2,p=probabilities_2)
   227       846 2514167007.0 2971828.6     10.5                  vec=op_list[op_2](vec)
   228       846  107827797.0 127456.0      0.5                  self.update_history(vec,op_2)
```
2. Now the break down the three
2.1 Control_map:
```
Timer unit: 1e-09 s

Total time: 8.28536 s
File: /tmp/ipykernel_3549544/2655625021.py
Function: control_map at line 115

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   115                                               def control_map(self,vec,bL):
   116                                                   '''control map depends on the outcome of the measurement of bL'''
   117                                                   # projection on the last bits
   118                                                   # P_cached=P(self.L,bL)
   119                                                   # vec=P_cached@vec
   120       377 1848203438.0 4902396.4     22.3          vec=self.P_tensor(vec,bL)
   121       195     160588.0    823.5      0.0          if bL==1:
   122       195 1152885744.0 5912234.6     13.9              vec=self.XL_tensor(vec)
   123       377 1856688030.0 4924901.9     22.4          vec=normalize(vec)
   124                                                   # right shift 
   125                                                   # vec=T(self.L,left=False)@vec
   126       377  845989808.0 2244004.8     10.2          vec=self.T_tensor(vec,left=False)
   127                                           
   128       377  969009043.0 2570315.8     11.7          assert np.abs(vec[vec.shape[0]//2:]).sum() == 0, f'first qubit is not zero ({np.abs(vec[vec.shape[0]//2:]).sum()}) after right shift '
   129                                           
   130                                                   # Adder
   131       377     319544.0    847.6      0.0          if not self.ancilla:
   132       377 1611985569.0 4275823.8     19.5              vec=self.adder()@vec
   133                                                   else:
   134                                                       vec=(self.adder()@vec.reshape((2**self.L,2))).flatten()
   135                                                   
   136       377     118075.0    313.2      0.0          return vec
```
2.2 projection_map
```
Timer unit: 1e-09 s

Total time: 2.51573 s
File: /tmp/ipykernel_3549544/2655625021.py
Function: projection_map at line 138

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   138                                               def projection_map(self,vec,pos,n):
   139                                                   '''projection to `pos` with outcome of `n`
   140                                                   note that here is 0-index, and pos=L-1 is the last bit'''
   141                                                   # vec=P(self.L,n=n,pos=pos)@vec
   142       393  652875317.0 1661260.3     26.0          vec=self.P_tensor(vec,n,pos)
   143       393 1862726202.0 4739761.3     74.0          vec=normalize(vec)
   144                                           
   145                                                   # proj to any axis
   146                                                   # U_2=U(2,self.rng)
   147                                                   # # if not self.ancilla:
   148                                                   # vec_tensor=vec.reshape((2,)*self.L_T)
   149                                                   # idx_list=np.arange(self.L_T)
   150                                                   # idx_list[pos],idx_list[0]=idx_list[0],idx_list[pos]
   151                                                   # vec_tensor=vec_tensor.transpose(idx_list).reshape((2,2**(self.L_T-1)))
   152                                                   # vec=(U_2@vec_tensor).reshape((2,)*self.L_T).transpose(idx_list).flatten()
   153                                           
   154       393     126651.0    322.3      0.0          return vec
   ```
2.3 Bernoulli_map
```
Timer unit: 1e-09 s

Total time: 5.91561 s
File: /tmp/ipykernel_3549544/2655625021.py
Function: Bernoulli_map at line 108

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   108                                               def Bernoulli_map(self,vec):
   109                                                   # vec=T(self.L,left=True)@vec
   110       423 4284375934.0 10128548.3     72.4          vec=self.T_tensor(vec,left=True)
   111                                                   # vec=S(self.L,rng=self.rng)@vec
   112       423 1631005101.0 3855804.0     27.6          vec=self.S_tensor(vec,rng=self.rng)
   113       423     230387.0    544.7      0.0          return vec
```