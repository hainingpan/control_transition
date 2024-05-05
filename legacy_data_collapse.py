
## Legacy data collapse code
# import torch
# class Optimizer:
#     def __init__(self,p_c,nu,df,params={'Metrics':'O',},p_range=[-0.1,0.1],Lmin=None,Lmax=None,bootstrap=False,gaussian_check=False):
#         self.p_c=torch.tensor([p_c],requires_grad=False)
#         self.nu=torch.tensor([nu],requires_grad=False)
#         self.p_range=p_range
#         self.Lmin=0 if Lmin is None else Lmin
#         self.Lmax=100 if Lmax is None else Lmax
#         self.bootstrap=bootstrap
#         self.gaussian_check=gaussian_check
#         self.params=params
#         self.df=self.load_dataframe(df,params)
#         self.L_i,self.p_i,self.d_i,self.y_i = self.load_data()

    
#     def load_dataframe(self,df,params):
#         import scipy

#         df=df.xs(params.values(),level=list(params.keys()))['observations']
#         df=df[(df.index.get_level_values('p')<=self.p_c.item()+self.p_range[1]) & (self.p_c.item()+self.p_range[0]<=df.index.get_level_values('p'))]
#         df=df[(df.index.get_level_values('L')<=self.Lmax) & (self.Lmin<=df.index.get_level_values('L'))]
#         if self.bootstrap:
#             df=df.apply(lambda x: list(np.random.choice(x,size=len(x),replace=True)))
#         if self.gaussian_check:
#             print(df.apply(scipy.stats.shapiro))
#         return df
    
#     def load_data(self):
#         L_i=torch.from_numpy(self.df.index.get_level_values('L').values)
#         p_i=torch.from_numpy(self.df.index.get_level_values('p').values)
#         d_i=torch.from_numpy(self.df.apply(np.std).values)/np.sqrt(self.df.apply(len).values)
#         y_i=torch.from_numpy(self.df.apply(np.mean).values)
#         assert p_i.unique().shape[0]>=4, f'not enough data points {p_i.unique().shape[0]}'
#         return L_i,p_i,d_i,y_i

#     def loss(self,p_c,nu,MLE=True):
#         x_i=(self.p_i-p_c)*(self.L_i)**(1/nu)
#         order=x_i.argsort()
#         x_i_ordered=x_i[order]
#         y_i_ordered=self.y_i[order]
#         d_i_ordered=self.d_i[order]
#         x={i:x_i_ordered[1+i:x_i_ordered.shape[0]-1+i] for i in [-1,0,1]}
#         d={i:d_i_ordered[1+i:d_i_ordered.shape[0]-1+i] for i in [-1,0,1]}
#         y={i:y_i_ordered[1+i:y_i_ordered.shape[0]-1+i] for i in [-1,0,1]}
#         x_post_ratio=(x[1]-x[0])/(x[1]-x[-1])
#         x_pre_ratio=(x[-1]-x[0])/(x[1]-x[-1])
#         y_var=d[0]**2+(x_post_ratio*d[-1])**2+(x_pre_ratio*d[1])**2
#         y_bar=x_post_ratio*y[-1]-x_pre_ratio*y[1]
#         # return torch.sum((y[0]-y_bar)**2/y_var)
#         if MLE:
#             return self.MLE(y[0],y_bar,y_var)
#         else:
#             return self.chi2(y[0],y_bar,y_var)
    
#     def loss_drift(self,p_c,nu,omega,beta):
#         x_i=(self.p_i-p_c)*(self.L_i)**(1/nu)
#         y_var=self.d_i**2
#         self.y_i_fitted=0
#         for i,b in enumerate(beta[:-1]):
#             self.y_i_fitted+=b*x_i**i
#         self.y_i_fitted+=beta[-1]/self.L_i**omega
#         self.p_c=p_c
#         self.nu=nu
#         self.omega=omega
#         self.beta=beta
        
#         # self.y_i_fitted=a+b*x_i+c*x_i**2+d/self.L_i**omega
#         return self.chi2(self.y_i_fitted,self.y_i,y_var)

    
#     def loss_drift_sample(self,p_c,nu,omega,b1,b2,a):
#         """return the residual of each sample, 
#         a.shape(n1+1,n2+1), a=[[a00, a01=1, a02, a03, ...],
#                                [a10=1, a11, a12, a13, ...],
#                                [a20, a21,   a22, a23, ...],
#                                ...]
#         b1.shape=(m1,), b1=[b10=0,b11,b12,...]
#         b2.shape=(m2+1,), b2=[b20,b21,b22,...]
#         w_i.shape=(n_sample,)"""
#         w_i=((self.p_i-p_c)/p_c)    # (n_sample,)
#         u1_i=torch.tensor(b1)@w_i**torch.arange(1,b1.shape[0]+1)[:,None]    # (n_sample,) because b10=0 to ensure u1(w=0)=0
#         u2_i=torch.tensor(b2)@w_i**torch.arange(b2.shape[0])[:,None]  # (n_sample,)
#         phi_1=u1_i*(self.L_i)**(1/nu)    # (n_sample,)
#         phi_2=u2_i*(self.L_i)**(-omega)  # (n_sample,)
#         phi_1_=phi_1 ** torch.arange(a.shape[0])[:,None]    # (n1+1,n_sample)
#         phi_2_=phi_2 ** torch.arange(a.shape[1])[:,None]    # (n2+1,n_sample)
#         self.y_i_fitted=torch.einsum('ij,ik,kj->j',phi_1_,torch.tensor(a),phi_2_)


#         self.p_c=p_c
#         self.nu=nu
#         self.omega=omega
        
#         return (self.y_i_fitted-self.y_i)/self.d_i

#     def chi2(self,y,y_fitted,sigma2):
#         return 0.5*torch.sum((y-y_fitted)**2/sigma2)
    
#     def MLE(self,y,y_fitted,sigma2):
#         return 0.5*torch.sum((y-y_fitted)**2/sigma2)+0.5*torch.sum(torch.log(sigma2))

    
#     def visualize(self,p_c_range,nu_range,trajectory=False,fig=True,ax=None,mapfunc=lambda x:x):
#         from matplotlib.colors import LogNorm
#         import matplotlib.pyplot as plt
#         from matplotlib.colors import LogNorm
#         p_c_list=np.linspace(*p_c_range,82)
#         nu_list=np.linspace(*nu_range,80)
#         loss_map=np.array([[self.loss(torch.tensor([p_c]),torch.tensor([nu]),MLE=False).item() for p_c in p_c_list] for nu in nu_list])
#         if fig:
#             if ax is None:
#                 fig, ax = plt.subplots()
#             cm=ax.contourf(p_c_list,nu_list,mapfunc(loss_map),levels=20)
#             ax.set_xlabel(r'$p_c$')
#             ax.set_ylabel(r'$\nu$')
#             plt.colorbar(cm)
#             if trajectory:
#                 ax.scatter(self.p_c_history,self.nu_history,s=np.linspace(3,1,len(self.p_c_history))**2,)
#             ct=ax.contour(p_c_list,nu_list,mapfunc(loss_map),levels=[mapfunc(self.loss(self.p_c,self.nu,MLE=False).item()*1.3),],colors='k',linestyles='dashed')
#         else:
#             ct=plt.contour(p_c_list,nu_list,mapfunc(loss_map),levels=[mapfunc(self.loss(self.p_c,self.nu,MLE=False).item()*1.3),],colors='k',linestyles='dashed');
#         params_range=ct.collections[0].get_paths()[0].vertices
#         return params_range[:,0].min(),params_range[:,0].max(),params_range[:,1].min(),params_range[:,1].max()

#     def optimize(self,tolerance=1e-10):
#         """Optimize using pytorch, Gradient Descent method"""
#         p_c_prime = torch.tensor([torch.logit(self.p_c)],requires_grad=True)
#         nu_prime = torch.tensor([torch.log(self.nu)],requires_grad=True)
#         optimizer=torch.optim.Adam([p_c_prime,nu_prime],)
#         # optimizer=torch.optim.Adam([self.p_c,self.nu],)
#         prev_loss=float('inf')
#         current_loss=0
#         self.loss_history=[]
#         # self.p_c_history=[self.p_c.item()]
#         # self.nu_history=[self.nu.item()]
#         self.p_c_history=[torch.sigmoid(p_c_prime).item()]
#         self.nu_history=[torch.exp(nu_prime).item()]
#         iteration=0
#         while abs(prev_loss-current_loss)>tolerance and iteration<10000:
#             p_c_transformed = torch.sigmoid(p_c_prime)
#             nu_transformed = torch.exp(nu_prime)

#             loss_ = self.loss(p_c_transformed, nu_transformed,MLE=False)
#             # loss_=self.loss(self.p_c,self.nu)
#             optimizer.zero_grad()
#             loss_.backward()
#             optimizer.step()
#             prev_loss=current_loss
#             current_loss=loss_.item()
#             self.loss_history.append(current_loss)
#             self.p_c_history.append(p_c_transformed.item())
#             self.nu_history.append(nu_transformed.item())
#             # self.p_c_history.append(self.p_c.item())
#             # self.nu_history.append(self.nu.item())
#             iteration+=1
#         self.p_c = torch.sigmoid(p_c_prime)
#         self.nu = torch.exp(nu_prime)
#         Hessian= torch.tensor(torch.autograd.functional.hessian(self.loss,(self.p_c,self.nu)))
#         self.se=torch.sqrt(torch.diag(torch.inverse(Hessian)))
        
#         return {'p_c':self.p_c.item(),'nu':self.nu.item(),'loss':current_loss*2/(self.y_i.shape[0]-2),'se':self.se.detach().numpy()}

#     def optimize_scipy(self):
#         """Optimize using scipy.minimize"""
#         import scipy

#         func=lambda x: self.loss(torch.tensor([x[0]]),torch.tensor([x[1]]),MLE=False).item()
#         res=scipy.optimize.minimize(func,[self.p_c.item(),self.nu.item()],method='Nelder-Mead',bounds=[(0,1),(0,2)])
#         # res=scipy.optimize.minimize(func,[self.p_c.item(),self.nu.item()],method='L-BFGS-B',bounds=[(0,1),(0,5)])
#         # 'L-BFGS-B',bounds=[(0,1),(0,5)]
#         Hessian= torch.tensor(torch.autograd.functional.hessian(self.loss,(torch.tensor(res.x[0]),torch.tensor(res.x[1]))))
#         se=torch.sqrt(torch.diag(torch.inverse(Hessian)))
#         self.p_c=torch.tensor([res.x[0]])
#         self.nu=torch.tensor([res.x[1]])
#         return res,res.fun*2/(self.y_i.shape[0]-2),se

#     def optimize_drift(self,omega,a,b,c,d,tolerance=1e-10,):
#         """Optimize using pytorch, Gradient Descent method with consideration of drifting of crossing point. Scaling function is Talyor expansion in  PHYS. REV. X 12, 041002 (2022)"""
#         p_c_prime = torch.tensor([torch.logit(self.p_c)],requires_grad=True)
#         nu_prime = torch.tensor([torch.log(self.nu)],requires_grad=True)
#         omega=torch.tensor([omega],requires_grad=True,dtype=torch.float32)
#         a=torch.tensor([a],requires_grad=True,dtype=torch.float32)
#         b=torch.tensor([b],requires_grad=True,dtype=torch.float32)
#         c=torch.tensor([c],requires_grad=True,dtype=torch.float32)
#         d=torch.tensor([d],requires_grad=True,dtype=torch.float32)
#         optimizer=torch.optim.Adam([p_c_prime,nu_prime,omega,a,b,c,d],)
#         prev_loss=float('inf')
#         current_loss=0
#         self.loss_history=[]
#         self.p_c_history=[torch.sigmoid(p_c_prime).item()]
#         self.nu_history=[torch.exp(nu_prime).item()]
#         iteration=0
#         while abs(prev_loss-current_loss)>tolerance and iteration<100000:
#             p_c_transformed = torch.sigmoid(p_c_prime)
#             nu_transformed = torch.exp(nu_prime)

#             loss_ = self.loss_drift(p_c_transformed, nu_transformed,omega,a,b,c,d)
#             optimizer.zero_grad()
#             loss_.backward()
#             optimizer.step()
#             prev_loss=current_loss
#             current_loss=loss_.item()
#             self.loss_history.append(current_loss)
#             self.p_c_history.append(p_c_transformed.item())
#             self.nu_history.append(nu_transformed.item())
#             iteration+=1
#         self.p_c = torch.sigmoid(p_c_prime)
#         self.nu = torch.exp(nu_prime)
#         return {'p_c':self.p_c.item(),'nu':self.nu.item(),'omega':omega.item(),'a':a.item(),'b':b.item(),'c':c.item(),'d':d.item(),'loss':current_loss,'chi-square_nu':current_loss*2/(self.y_i.shape[0]-7)}

#     def optimize_drift_scipy(self,omega,a,b,c,d):
#         """Optimize using scipy.minimize, using taylor expansion PHYS. REV. X 12, 041002 (2022)"""
#         # omega,a,b,c,d=
#         import scipy

#         func=lambda x: self.loss_drift(*tuple(x),d=d).item()
#         res=scipy.optimize.minimize(func,[self.p_c.item(),self.nu.item(),omega,a,b,c],method='Nelder-Mead')
#         Hessian= torch.tensor(torch.autograd.functional.hessian(lambda x: self.loss_drift(*x,d=d),torch.tensor(res.x)))
#         # se=torch.sqrt(torch.diag(torch.inverse(Hessian)))
#         # self.p_c=torch.tensor([res.x[0]])
#         # self.nu=torch.tensor([res.x[1]])
#         return res,res.fun*2/(self.y_i.shape[0]-7),

#     def linear_least_square(self,p_c,nu,omega,n):
#         """n is the order of relevant parts"""
#         X=torch.zeros((self.y_i.shape[0],n+2),dtype=torch.float64)
#         X[:,0]=torch.ones_like(self.y_i)
#         x_i=(self.p_i-p_c)*(self.L_i)**(1/nu)
#         for j in range(1,n+1):
#             X[:,j]=x_i**j
#         X[:,n+1]=1/self.L_i**omega
#         Y=self.y_i
#         Sigma_inv=torch.diag(1/self.d_i**2)
#         XY=X.T @ Sigma_inv @ Y
#         XX=X.T @ Sigma_inv @ X
#         # beta=torch.linalg.solve(XX,XY)
#         beta=torch.linalg.inv(XX)@XY
#         return beta
#     def optimize_drift_lsq(self,omega,n=2):
#         """generalized version of PHYS. REV. X 12, 041002 (2022), to n-th order"""
#         import scipy

#         def func(x):
#             beta= self.linear_least_square(p_c=x[0],nu=x[1],omega=x[2],n=n)
#             return self.loss_drift(p_c=x[0],nu=x[1],omega=x[2],beta=beta).item()

#         res=scipy.optimize.minimize(func,[self.p_c.item(),self.nu.item(),omega],method='Nelder-Mead',bounds=[(min(self.p_i),max(self.p_i)),(.2,3),(1e-4,None)],)
#         self.p_c=torch.tensor([res.x[0]])
#         self.nu=torch.tensor([res.x[1]])
#         return res, res.fun*2/(self.y_i.shape[0]-3), 

#     def optimize_drift_nonlinear_lsq(self,omega, m1,m2,n1,n2,x0=None):
#         """m1, m2 controls the order of (p-p_c)/p_c, in the relevant and irrelevant operaor.
#         n1, n2 controls the order of phi=u((p-p_c)/p_c) * L^{1/nu} and  phi=u((p-p_c)/p_c) * L^{-omega}   
#         n2 can be zero while n1 cannot be zero, 

#         TODO: the mixing of torch and numpy is messy, should use a clean version

#         """ 
#         import scipy

#         assert n1>0, 'n1 should be greater than 0'

#         def func(x):
#             p_c,nu,omega=x[0],x[1],x[2]
#             b1=x[3:3+m1]
#             if n2>0:
#                 b2=x[3+m1:3+m1+m2+1]
#             else:
#                 b2=np.array([])
#             a=torch.zeros((n1+1,n2+1),dtype=torch.float64)
#             ## This is not correct, why m2 is there when n2 is zero?
#             a[0,0]=x[3+m1+m2+1]
#             if n2>0:
#                 a[0,1]=1
#                 if n2>1:
#                     a[0,2:n2+1]=torch.tensor([x[3+m1+m2+2:3+m1+m2+2+n2-1]])
#             if n1>0:
#                 a[1,0]=1
#                 a[1,1:n2+1]=torch.tensor(x[3+m1+m2+2+n2-1:3+m1+m2+2+n2-1+n2])
#                 if n1>1:
#                     a[2:,:]=torch.tensor(x[3+m1+m2+2+n2-1+n2:].reshape(n1-1,n2+1))

#             self.x0=x

#             return self.loss_drift_sample(p_c,nu,omega,b1,b2,a)
#         if x0 is None:
#             x0=[self.p_c.item(),self.nu.item(),omega]+[0]*(m1+m2+1+(n1+1)*(n2+1)-2)
#         else:
#             x0=[self.p_c.item(),self.nu.item(),omega]+x0
#         res=scipy.optimize.least_squares(func,x0,method='lm',)
#         return res, res.cost*2/(self.y_i.shape[0]-len(x0))



#     def plot_loss(self):
#         import matplotlib.pyplot as plt
#         if hasattr(self, 'loss_history'):
#             fig,ax=plt.subplots()
#             ax.plot(self.loss_history,'.-')
#             ax.set_xlabel('Iteration')
#             ax.set_ylabel('O')
    
#     def plot_data_collapse(self,ax=None,drift=False):
#         from matplotlib.colors import LogNorm
#         import matplotlib.pyplot as plt
#         from matplotlib.colors import LogNorm
#         x_i=(self.p_i-self.p_c)*(self.L_i)**(1/self.nu)
#         # x_i=self.p_i
#         if ax is None:
#             fig,ax = plt.subplots()
#         L_list=self.df.index.get_level_values('L').unique().sort_values().values
#         idx_list=[0]+(np.cumsum([self.df.xs(key=L,level='L').shape[0] for L in L_list])).tolist()
#         L_dict={L:(start_idx,end_idx) for L,start_idx,end_idx in zip(L_list,idx_list[:-1],idx_list[1:])}
#         # color_iter=iter(plt.cm.rainbow(np.linspace(0,1,len(L_list))))
#         color_iter = iter(plt.cm.Blues(0.4+0.6*(i/L_list.shape[0])) for i in range(L_list.shape[0]))
#         for L,(start_idx,end_idx) in L_dict.items():
#             color=next(color_iter)
#             if drift:
#                 ax.errorbar(self.p_i.detach().numpy()[start_idx:end_idx], self.y_i.detach().numpy()[start_idx:end_idx], label=f'{L}', color=color, yerr=self.d_i.detach().numpy()[start_idx:end_idx], capsize=2, fmt='x',linestyle="None")
#                 ax.plot(self.p_i.detach().numpy()[start_idx:end_idx],self.y_i_fitted.detach().numpy()[start_idx:end_idx],label=f'{L}',color=color)
                
#             else:
#                 ax.scatter(x_i.detach().numpy()[start_idx:end_idx],self.y_i.detach().numpy()[start_idx:end_idx],label=f'{L}',color=color)
#                 # ax.plot(x_i.detach().numpy()[start_idx:end_idx],self.y_i_fitted.detach().numpy()[start_idx:end_idx],label=f'{L}')
                


#         if drift:
#             ax.set_xlabel(r'$p_i$')
#             ax.set_title(rf'$p_c={self.p_c.item():.3f},\nu={self.nu.item():.3f},\omega = {self.omega.item():.3f}$')
#         else:
#             ax.set_xlabel(r'$(p_i-p_c)L^{1/\nu}$')
#             ax.set_title(rf'$p_c={self.p_c.item():.3f},\nu={self.nu.item():.3f}$')
#         ax.set_ylabel(r'$y_i$')
#         ax.legend()
#         ax.grid('on')

#         # adder=self.df.index.get_level_values('adder').unique().tolist()[0]
#         # print(f'{self.params["Metrics"]}_Scaling_L({L_list[0]},{L_list[-1]})_adder({adder[0]}-{adder[1]}).png')
        
    
#     def plot_line(self):
#         import matplotlib.pyplot as plt
#         fig,ax=plt.subplots()
#         ax.plot(self.p_i,self.y_i)


---

# def add_optimal(optimal_model,model,names=['Metric', 'p_proj', 'p_ctrl']):
#     import pandas as pd
#     df_new = pd.DataFrame([model])
#     p_c_key=frozenset(names)-frozenset(model.params.keys())

#     index_list=[]
#     for name in names:
#         if name in model.params:
#             index_list.append(model.params[name])
#         else:
#             index_list.append(None)
#     index = pd.MultiIndex.from_tuples([tuple(index_list)],names=names)
#     p_c=model.res.params['p_c'].value
#     p_c_error=model.res.params['p_c'].stderr
#     nu=model.res.params['nu'].value
#     nu_error=model.res.params['nu'].stderr
#     if 'y' in model.res.params:
#         y=model.res.params['y'].value
#         y_error=model.res.params['y'].stderr
#     else:
#         y=None
#         y_error=None
#     new={
#         'p_c':p_c, 
#         'p_c_error':p_c_error,
#         'nu': nu,
#         'nu_error': nu_error,
#         'y': y,
#         'y_error': y_error}
#     new_df=pd.DataFrame(new,index=index)
#     # return  new_df
#     return pd.concat([optimal_model,new_df],axis=0)

# def initialize_df(names=['Metric', 'p_proj', 'p_ctrl']):
#     import pandas as pd
#     return pd.DataFrame(
#     columns=['p_c', 'p_c_error', 'nu', 'nu_error', 'y', 'y_error'],
#     index= pd.MultiIndex(levels=[[]]*len(names), codes=[[]]*len(names), names=names)
# )