import numpy as np
import matplotlib.pyplot as plt
class Diffusion:
    diffusion_type_ylabel={
    1: r'$\mathbb{E}_m[\langle{\hat{F}}\rangle^2]- \mathbb{E}_m[\langle{\hat{F}}\rangle]^2$',
    2: r'$\mathbb{E}_m[\langle(\hat{F}-\langle\hat{F}\rangle)^2\rangle]$',
    3: r'$\mathbb{E}_m[\langle(\hat{F}-\langle\hat{F}\rangle)^2\rangle^2]^{1/2}$',
    4: r'$[\mathbb{E}_m[\langle(\hat{F}-\langle\hat{F}\rangle)^2\rangle^2]-\mathbb{E}_m[\langle(\hat{F}-\langle\hat{F}\rangle)^2\rangle]^2]^{1/2}$'
    }
    diffusion_xlabel={1: r'$D$', 2: r'$D_2$', 3: r'$D_3$', 4: r'$\sqrt{D_3^2-D_2^2}$'}
    diffusion_type_str= {1:'var_mean',2: 'mean_var', 3:'mean_var2', 4:'std_var'}
    diffusion_rate={1:'D',2:'D2',3:'D3',4:'D32'}
    fixed_str = {'AFM':'1_3', 'FM':'0'}
    p_label={True:'ctrl',False:'proj'}
    t_label={None:r'$t$','sqrt':r'$\sqrt{t}/L$','log':r'$\log{t/L}$','linear':r'$t/L$','sqrt2':r'$\sqrt{t/L}$'}
    t_func_={'sqrt':lambda t,L:np.sqrt(t)/L,'log':lambda t,L:np.log(t/L),'linear':lambda t,L:t/L,'sqrt2':lambda t,L:np.sqrt(t/L)}

    t_func_fn={None:'_t', 'sqrt':'_sqrt', 'log':'_log', 'linear':'_linear','sqrt2':'_sqrt2'}

    def __init__(self,fdw,fixed,diffusion_type,ctrl):
        self.diffusion_type=diffusion_type
        self.ctrl=ctrl
        self.fixed=fixed
        self.fdw=fdw[(fixed,self.p_label[ctrl])]
    
    def generate_data(self,L,idx,func):
        '''L : length of system
        idx: index of p_list or p_list'''
        t=np.arange(2*L**2+1)
        if func is not None:
            x=self.t_func_[func](t,L)
        else:
            x=t
        slicing=(idx,0) if self.ctrl else (0,idx)
        if self.diffusion_type==1:
            y=np.var(self.fdw[L][f'FDW_{L}'][slicing[0],slicing[1],:,0,:,0],axis=-1)
        elif self.diffusion_type==2:
            var_state=self.fdw[L][f'FDW_{L}'][slicing[0],slicing[1],:,1,:,0]-self.fdw[L][f'FDW_{L}'][slicing[0],slicing[1],:,0,:,0]**2
            y=np.mean(var_state,axis=-1)
        elif self.diffusion_type==3:
            var_state=self.fdw[L][f'FDW_{L}'][slicing[0],slicing[1],:,1,:,0]-self.fdw[L][f'FDW_{L}'][slicing[0],slicing[1],:,0,:,0]**2
            y=np.mean(var_state**2,axis=-1)**0.5
        elif self.diffusion_type==4:
            var_state=self.fdw[L][f'FDW_{L}'][slicing[0],slicing[1],:,1,:,0]-self.fdw[L][f'FDW_{L}'][slicing[0],slicing[1],:,0,:,0]**2
            y=np.sqrt(np.mean(var_state**2,axis=-1)-np.mean(var_state,axis=-1)**2)
        return x,y

    def plot_fit(self,L_list,p_list,fit,idx_list,xlim_func,L_alpha,t_func=None,shift_step=0,xlim=(0,20),markersize=3,lw=2,ax=None,fmt='.',color_list=['C0','C1','C2','C3','C6']):
        if ax is None:
            fig,ax=plt.subplots(figsize=(6,5))
        alpha_list=np.linspace(1,0.4,len(L_list))
        
        
        if L_alpha:
            outer_list, inner_list = list(zip(enumerate(idx_list),color_list)), list(zip(enumerate(L_list),alpha_list))
        else:
            outer_list, inner_list = list(zip(enumerate(L_list),color_list)), list(zip(enumerate(idx_list),alpha_list))
        for outer in outer_list:
            for inner in inner_list:
                if L_alpha:
                    (shift_idx,idx),color=outer
                    (L_idx,L),alpha=inner
                    shift=shift_idx*shift_step
                else:
                    (L_idx,L),color=outer
                    (shift_idx,idx),alpha=inner
                    shift=L_idx*shift_step
                x,y=self.generate_data(L,idx,t_func)
                if t_func is None:
                    ax.plot(shift+y,fmt,label=f'$p_{{{self.p_label[self.ctrl]}}}$={p_list[idx]:.2f}, $L$={L}',color=color,alpha=alpha,markersize=markersize,linewidth=lw)
                else:
                    ax.plot(x,shift+y,label=f'$p_{{{self.p_label[self.ctrl]}}}$={p_list[idx]:.2f}, $L$={L}',color=color,alpha=alpha,linewidth=lw)
                if fit:
                    # slope=self.fit(idx,L,y)
                    fit_xlim=xlim_func[idx](L)
                    slope=diffusion_coef_fit(y,fit_xlim)
                    plot_xlim=np.arange(fit_xlim[0]-1,(fit_xlim[1]+4))
                    ax.plot(
                        plot_xlim,
                        shift+slope[0]*plot_xlim+slope[1],
                        color=color,
                        alpha=alpha,
                        ls='dashed',
                        lw=.5)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=8)
        ax.set_xlabel(self.t_label[t_func])
        
        ax.set_ylabel(self.diffusion_type_ylabel[self.diffusion_type])
        ax.grid('on')
        save_string = self.diffusion_type_str[self.diffusion_type] + '_'+self.fixed_str[self.fixed] + f'_{self.p_label[self.ctrl]}'+ self.t_func_fn[t_func]+'.png'
        return save_string

    def fit(self,xlim_func,L_list):
        return {L:np.array([diffusion_coef_fit(self.generate_data(L,p_idx,None)[1],xlim_func[p_idx](L)) for p_idx in range(21)]) for L in L_list} 
    
    def EE(self,L_list):
        if self.ctrl:
            return {L:[np.mean(self.fdw[L]['EE'][p_idx,0]) for p_idx in range(20)] for L in L_list}
        else:
            return {L:[np.mean(self.fdw[L]['EE'][0,p_idx]) for p_idx in range(20)] for L in L_list}

    def plot_D_L(self,p_list,p_idx_list,L_list,diffusion):
        fig,ax=plt.subplots(figsize=(4,3))
        for p_idx in p_idx_list:
            ax.plot(1/np.array(L_list),[diffusion[L][p_idx,0] for L in L_list],'.-',label=f'$p_{{{self.p_label[self.ctrl]}}}={p_list[p_idx]:.2f}$')
        ax.set_xticks(1/np.array(L_list),labels=[f'$\\frac{{1}}{{{L}}}$' for L in L_list])
        ax.legend()
        ax.set_xlabel('1/L')
        ax.set_ylabel(self.diffusion_xlabel[self.diffusion_type])
        ax.grid('on')
        save_string = self.diffusion_rate[self.diffusion_type] + '_L_'+self.fixed_str[self.fixed] + f'_{self.p_label[self.ctrl]}'+ '.png'
        return save_string
    
    def plot_D(self,p_list,diffusion,L=16,analytic=True,ax=None,label=None,legend=True,color='k',lw=1,markersize=1):
        if ax is None:
            fig,ax=plt.subplots(figsize=(4,3))
        ax.errorbar(p_list,[diffusion[L][p_idx,0] for p_idx in range(len(p_list))],yerr=[diffusion[L][p_idx,2] for p_idx in range(len(p_list))],fmt='.-',label=f'$L=${L}' if label is None else label,capsize=1,capthick=.5,color=color,lw=lw,markersize=1)
        if analytic:
            ax.plot(p_list,4*p_list*(1-p_list),label='$4p(1-p)$')
        ax.set_xlabel(f'$p_\\text{{{self.p_label[self.ctrl]}}}$')
        ax.set_ylabel(self.diffusion_xlabel[self.diffusion_type])
        if legend:
            ax.legend()
        ax.grid('on')
        save_string = self.diffusion_rate[self.diffusion_type] + '_RW_'+self.fixed_str[self.fixed] + f'_{self.p_label[self.ctrl]}'+ '.png'
        return save_string
    def plot_S_D(self,EE,diffusion):
        fig,ax=plt.subplots(figsize=(4,3))
        for L,color in zip(L_list,['C0','C1','C2','C3','C6']):
            ax.errorbar(diffusion[L][:,0],EE[L],xerr=diffusion[L][:,2],label=f'$L$={L}',capsize=3,color=color)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlabel(self.diffusion_xlabel[self.diffusion_type])
        ax.set_ylabel(r'$\overline{S_{L/2}}$')
        ax.grid('on')
        save_string = self.diffusion_rate[self.diffusion_type] + '_'+self.fixed_str[self.fixed] + f'_{self.p_label[self.ctrl]}'+ '.png'
        return save_string

    def plot_D_pctrl(self,p_list,L_list):
        fig,ax=plt.subplots(figsize=(6,5))
        for L in L_list:
            ax.plot(p_list,[self.generate_data(L=L,idx=idx,func=None)[1][-1] for idx in range(len(p_list))],'.-',label=f'L={L}',markersize=3)
        ax.grid('on')
        ax.legend()
        ax.set_xlabel('$p_{ctrl}$')
        ax.set_ylabel(self.diffusion_type_ylabel[self.diffusion_type])
        save_string = self.diffusion_type_str[self.diffusion_type] + '_'+self.fixed_str[self.fixed] + f'_pctrl'+ '.png'
        return save_string

def diffusion_coef(f_t,p_idx,L,p_proj=True):
    from scipy.stats import linregress
    if p_proj:
        y=np.var(f_t[L][f'FDW_{L}'],axis=-2)[0,p_idx,:L//2,0,0]
    else:
        y=np.var(f_t[L][f'FDW_{L}'],axis=-2)[p_idx,0,:L//2,0,0]
    x=np.arange(len(y))
    res=linregress(x,y)
    return res.slope,res.stderr

def diffusion_coef_C(f_t,dw,p_idx,L,):
    from scipy.stats import linregress
    y=np.var(dw[L][f_t[L][f'wf_{L}'][p_idx]],axis=-1)[:L//2]
    x=np.arange(len(y))
    res=linregress(x,y)
    return res.slope,res.stderr

def diffusion_coef_fit(y,xrange):
    from scipy.stats import linregress
    x=np.arange(len(y))[xrange[0]:xrange[1]+1]
    y=y[xrange[0]:xrange[1]+1]
    res=linregress(x,y)
    return res.slope, res.intercept, res.stderr