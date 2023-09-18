1. start, time `random_control_2`
---
Timer unit: 1e-06 s

Total time: 16.3872 s

Could not find file /tmp/ipykernel_5057/3905145613.py
Are you sure you are running this program from the same directory
that you ran the profiler from?
Continuing without the function's contents.

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   167                                           
   168                                           
   169                                           
   170       100       1128.0     11.3      0.0  
   171                                           
   172       100    3261191.0  32611.9     19.9  
   173                                           
   174       100        374.0      3.7      0.0  
   175       100        545.0      5.5      0.0  
   176                                           
   177       100      14164.0    141.6      0.1  
   178                                           
   179       200        760.0      3.8      0.0  
   180       100        146.0      1.5      0.0  
   181       100        382.0      3.8      0.0  
   182       100        163.0      1.6      0.0  
   183       100        210.0      2.1      0.0  
   184       100        191.0      1.9      0.0  
   185       100        115.0      1.1      0.0  
   186       100        153.0      1.5      0.0  
   187                                           
   188       100    8691730.0  86917.3     53.0  
   189       100       1030.0     10.3      0.0  
   190                                           
   191       100        161.0      1.6      0.0  
   192       144        226.0      1.6      0.0  
   193        96    4085921.0  42561.7     24.9  
   194        96        694.0      7.2      0.0  
   195        96        469.0      4.9      0.0  
   196        96      13081.0    136.3      0.1  
   197        96     313677.0   3267.5      1.9  
   198        96        698.0      7.3      0.0

2. break down
   control_map:   1.93891 s : 52
   projection_map: 0.290377s : 25
   Bernoulli_map: 5.17061 s: 48


2.3 let focus on Bernoulli_map first:
Timer unit: 1e-06 s

Total time: 4.60282 s

Could not find file /tmp/ipykernel_5057/3905145613.py
Are you sure you are running this program from the same directory
that you ran the profiler from?
Continuing without the function's contents.

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   358                                           
   359                                           
   360                                           
   361        48      12315.0    256.6      0.3  
   362        48      13423.0    279.6      0.3  
   363        48        145.0      3.0      0.0  
   364        48    4576941.0  95352.9     99.4

2.3.1 make recursive kron_list in S just appear one time, seems success
Timer unit: 1e-06 s

Total time: 0.063985 s

Could not find file /tmp/ipykernel_5057/2038999107.py
Are you sure you are running this program from the same directory
that you ran the profiler from?
Continuing without the function's contents.

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   358                                           
   359                                           
   360                                           
   361                                           
   362                                           
   363                                           
   364                                           
   365                                           
   366        48       9810.0    204.4     15.3  
   367        48      10991.0    229.0     17.2  
   368        48      43184.0    899.7     67.5

2.1 switch to control map
Timer unit: 1e-06 s

Total time: 1.93891 s

Could not find file /tmp/ipykernel_5057/2038999107.py
Are you sure you are running this program from the same directory
that you ran the profiler from?
Continuing without the function's contents.

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   102                                           
   103                                           
   104                                           
   105        52     575683.0  11070.8     29.7  
   106        52        233.0      4.5      0.0  
   107        29    1343885.0  46340.9     69.3  
   108        52       6171.0    118.7      0.3  
   109                                           
   110        52       6588.0    126.7      0.3  
   111                                           
   112        52       3266.0     62.8      0.2  
   113                                           
   114                                           
   115        52       3045.0     58.6      0.2  
   116                                           
   117        52         36.0      0.7      0.0

2.1.1 OK it does not seem to be something that I can do  

3. Go back to `random_control_2`.

Timer unit: 1e-06 s

Total time: 9.28917 s
File: /tmp/ipykernel_5057/3510496575.py
Function: random_control_2 at line 169

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   169                                               def random_control_2(self,p_ctrl,p_proj):
   170                                                   '''the competition between chaotic and random, where the projection can only be applied after the unitary
   171                                                   Notation: L-1 is the last digits'''
   172       100       1073.0     10.7      0.0          vec=self.vec_history[-1].copy()
   173                                                   
   174       100    2937476.0  29374.8     31.6          p= self.get_prob([self.L-1],vec)
   175                                           
   176       100        289.0      2.9      0.0          pool = ["C0","C1","chaotic"]
   177       100        542.0      5.4      0.0          probabilities = [p_ctrl * p[(self.L-1,0)], p_ctrl * p[(self.L-1,1)],  1- p_ctrl]
   178                                           
   179       100      14035.0    140.3      0.2          op = self.rng.choice(pool,p=probabilities)
   180                                           
   181       200        616.0      3.1      0.0          op_list= {"C0":partial(self.control_map,bL=0),
   182       100        118.0      1.2      0.0                    "C1":partial(self.control_map,bL=1),
   183       100        424.0      4.2      0.0                    f"P{self.L-1}0":partial(self.projection_map,pos=self.L-1,n=0),
   184       100        242.0      2.4      0.0                    f"P{self.L-1}1":partial(self.projection_map,pos=self.L-1,n=1),
   185       100        177.0      1.8      0.0                    f"P{self.L-2}0":partial(self.projection_map,pos=self.L-2,n=0),
   186       100        195.0      1.9      0.0                    f"P{self.L-2}1":partial(self.projection_map,pos=self.L-2,n=1),
   187       100        105.0      1.1      0.0                    "chaotic":self.Bernoulli_map,
   188       100        166.0      1.7      0.0                    "I":lambda x:x
   189                                                             }
   190       100    2082232.0  20822.3     22.4          vec=op_list[op](vec)
   191       100        815.0      8.2      0.0          self.update_history(vec,op)
   192                                           
   193       100        113.0      1.1      0.0          if op=="chaotic":
   194       144        182.0      1.3      0.0              for pos in [self.L-1,self.L-2]:
   195        96    3905938.0  40686.9     42.0                  p_2=self.get_prob([pos], vec)
   196        96        742.0      7.7      0.0                  pool_2=["I",f"P{pos}0",f"P{pos}1"]
   197        96        553.0      5.8      0.0                  probabilities_2=[1-p_proj, p_proj * p_2[(pos,0)], p_proj *  p_2[(pos,1)],]
   198        96      13861.0    144.4      0.1                  op_2 = self.rng.choice(pool_2,p=probabilities_2)
   199        96     328446.0   3421.3      3.5                  vec=op_list[op_2](vec)
   200        96        834.0      8.7      0.0                  self.update_history(vec,op_2)

3.1 Now get_prob is a problem, and this is because P(..) is very costly. An alternative way is to convert the vector to tensor (2,2,..), and then drag the index out, say j, transpose to (..., j), and reshape it into (2*(L-1), 2), [tensor[:,x].conj()@tensor[:,x] for x in range(2)]

Timer unit: 1e-06 s

Total time: 7.19143 s
File: /tmp/ipykernel_5057/3510496575.py
Function: get_prob at line 260

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   260                                               def get_prob(self,L_list,vec):
   261                                                   '''get the probability of measuring 0 at site L_list'''
   262                                                   # prob={(pos,n):(vec.conj().T@P(self.L,n=n,pos=pos)@vec).toarray()[0,0] for pos in L_list for n in [0,1]}
   263       196    7184769.0  36657.0     99.9          prob={(pos,n):(vec.conj().T@P(self.L,n=n,pos=pos)@vec) for pos in L_list for n in [0,1]}
   264       588       1435.0      2.4      0.0          for key, val in prob.items():
   265       392       4462.0     11.4      0.1              assert np.abs(val.imag)<self._eps, f'probability for {key} is not real {val}'
   266       392        633.0      1.6      0.0              prob[key]=val.real
   267       196        128.0      0.7      0.0          return prob


3.2 Now the time has reduced by a lot. 

Timer unit: 1e-06 s

Total time: 0.026036 s

Could not find file /tmp/ipykernel_5057/350946310.py
Are you sure you are running this program from the same directory
that you ran the profiler from?
Continuing without the function's contents.

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   269                                           
   270       196      24336.0    124.2     93.5  
   271       196       1512.0      7.7      5.8  
   272       196        188.0      1.0      0.7

4. Go back to `random_control_2`. The bottle neck become back to the `op`

Timer unit: 1e-06 s

Total time: 2.63872 s

Could not find file /tmp/ipykernel_5057/350946310.py
Are you sure you are running this program from the same directory
that you ran the profiler from?
Continuing without the function's contents.

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   169                                           
   170                                           
   171                                           
   172       100       1036.0     10.4      0.0  
   173                                           
   174       100      14958.0    149.6      0.6  
   175                                           
   176       100        172.0      1.7      0.0  
   177       100        282.0      2.8      0.0  
   178                                           
   179       100       9608.0     96.1      0.4  
   180                                           
   181       200        568.0      2.8      0.0  
   182       100        138.0      1.4      0.0  
   183       100        395.0      4.0      0.0  
   184       100        175.0      1.8      0.0  
   185       100        157.0      1.6      0.0  
   186       100        175.0      1.8      0.0  
   187       100         99.0      1.0      0.0  
   188       100        108.0      1.1      0.0  
   189                                           
   190       100    2230447.0  22304.5     84.5  
   191       100        707.0      7.1      0.0  
   192                                           
   193       100        130.0      1.3      0.0  
   194       144        217.0      1.5      0.0  
   195        96       9523.0     99.2      0.4  
   196        96        433.0      4.5      0.0  
   197        96        247.0      2.6      0.0  
   198        96       7976.0     83.1      0.3  
   199        96     360639.0   3756.7     13.7  
   200        96        532.0      5.5      0.0


5. I want to do a break down measurement again.
   control_map:   2.89802 s : 52
   projection_map: 0.328107 s : 25
   Bernoulli_map: 0.09105 s: 48

5.1 Again control_map is the bottleneck. So I will reduce it. now it is reduced greatly

Timer unit: 1e-06 s

Total time: 0.013011 s

Could not find file /tmp/ipykernel_5057/1386076930.py
Are you sure you are running this program from the same directory
that you ran the profiler from?
Continuing without the function's contents.

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   103                                           
   104                                           
   105                                           
   106                                           
   107                                           
   108        52       1589.0     30.6     12.2  
   109        52         72.0      1.4      0.6  
   110        29       1234.0     42.6      9.5  
   111        52       3100.0     59.6     23.8  
   112                                           
   113                                           
   114        52       1244.0     23.9      9.6  
   115                                           
   116        52       1880.0     36.2     14.4  
   117                                           
   118                                           
   119        52       3841.0     73.9     29.5  
   120                                           
   121        52         51.0      1.0      0.4

6.2 Now we optimize `projection_map`
Timer unit: 1e-06 s

Total time: 0.002046 s

Could not find file /tmp/ipykernel_5057/1847900001.py
Are you sure you are running this program from the same directory
that you ran the profiler from?
Continuing without the function's contents.

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   123                                           
   124                                           
   125                                           
   126                                           
   127        25        787.0     31.5     38.5  
   128        25       1243.0     49.7     60.8  
   129                                           
   130        25         16.0      0.6      0.8

6.3 For Bernoulli_map, we can also optimize T

Timer unit: 1e-06 s

Total time: 0.01549 s

Could not find file /tmp/ipykernel_5057/1728704479.py
Are you sure you are running this program from the same directory
that you ran the profiler from?
Continuing without the function's contents.

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    97                                           
    98                                           
    99        48       1635.0     34.1     10.6  
   100                                           
   101        48      13787.0    287.2     89.0  
   102        48         68.0      1.4      0.4


7. Run `random_control_2`, now the speed up is 16.3872/0.053205~ 308 times

Timer unit: 1e-06 s

Total time: 0.053205 s

Could not find file /tmp/ipykernel_5057/164457762.py
Are you sure you are running this program from the same directory
that you ran the profiler from?
Continuing without the function's contents.

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   173                                           
   174                                           
   175                                           
   176       100       1321.0     13.2      2.5  
   177                                           
   178       100       4395.0     44.0      8.3  
   179                                           
   180       100        122.0      1.2      0.2  
   181       100        192.0      1.9      0.4  
   182                                           
   183       100       5953.0     59.5     11.2  
   184                                           
   185       200        467.0      2.3      0.9  
   186       100        108.0      1.1      0.2  
   187       100        228.0      2.3      0.4  
   188       100        133.0      1.3      0.2  
   189       100        149.0      1.5      0.3  
   190       100        154.0      1.5      0.3  
   191       100        109.0      1.1      0.2  
   192       100        101.0      1.0      0.2  
   193                                           
   194       100      25883.0    258.8     48.6  
   195       100        485.0      4.8      0.9  
   196                                           
   197       100        137.0      1.4      0.3  
   198       144        181.0      1.3      0.3  
   199        96       4776.0     49.8      9.0  
   200        96        239.0      2.5      0.4  
   201        96        194.0      2.0      0.4  
   202        96       5335.0     55.6     10.0  
   203        96       2214.0     23.1      4.2  
   204        96        329.0      3.4      0.6

8. Further optimize ZZ, gives: 16.3872/0.014528 ~ 1000

Timer unit: 1e-06 s

Total time: 0.014528 s

Could not find file /tmp/ipykernel_976/275778611.py
Are you sure you are running this program from the same directory
that you ran the profiler from?
Continuing without the function's contents.

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   104                                           
   105                                           
   106                                           
   107                                           
   108                                           
   109        52       2477.0     47.6     17.0  
   110        52         64.0      1.2      0.4  
   111        29       1382.0     47.7      9.5  
   112        52       2899.0     55.8     20.0  
   113                                           
   114                                           
   115        52       1485.0     28.6     10.2  
   116                                           
   117        52       1998.0     38.4     13.8  
   118                                           
   119                                           
   120        52       4168.0     80.2     28.7  
   121                                           
   122        52         55.0      1.1      0.4

