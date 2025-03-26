import random, inspect
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
import cfpack as cfp
from cfpack import stop
import scipy.special as sp
np.seterr(divide='ignore', invalid='ignore')

#Global constants
N_len = 5
k_turb = r'$k_{\rm '+'turb'+'}$'; xlab = '$k$'#+k_turb
knu_label = r'$k_{\nu}$'
keta_label = r'$k_{\eta}$'
Pkin_label = r'$P_{\rm '+'kin'+'}$'
Pmag_label = r'$P_{\rm '+'mag'+'}$'

k_turb = 2
cons_nu_sub = 0.025
cons_nu_err_up_sub = 0.005
cons_nu_err_down_sub = 0.006
cons_nu_err_avg_sub = (cons_nu_err_up_sub + cons_nu_err_down_sub)/2

cons_nu_sup = cons_nu_sub * 0.33 / 0.1
err_sup = (0.01/0.33) + (0.01/0.1)
cons_nu_err_up_sup = cons_nu_sup * ((cons_nu_err_up_sub/cons_nu_sub)+err_sup)
cons_nu_err_down_sup = cons_nu_sup * ((cons_nu_err_down_sub/cons_nu_sub)+err_sup)
cons_nu_err_avg_sup = (cons_nu_err_up_sup + cons_nu_err_down_sup)/2

cons_eta_sub = 0.88
cons_eta_err_up_sub = 0.21
cons_eta_err_down_sub = 0.23
cons_eta_err_avg_sub = (cons_eta_err_up_sub + cons_eta_err_down_sub)/2

cons_eta_sup = cons_eta_sub * 0.33 / 0.53
err_sup = (0.03/0.33) + (0.07/0.53)
cons_eta_err_up_sup = cons_eta_sup * ((cons_eta_err_up_sub/cons_eta_sub)+err_sup)
cons_eta_err_down_sup = cons_eta_sup * ((cons_eta_err_down_sub/cons_eta_sub)+err_sup)
cons_eta_err_avg_sup = (cons_eta_err_up_sup + cons_eta_err_down_sup)/2

def cons(Mach,const,k_turb=k_turb):
    if Mach < 1:
        if const == 'nu': 
            return [cons_nu_sub*k_turb, cons_nu_err_avg_sub]
        elif const == 'eta': 
            return [cons_eta_sub, cons_eta_err_avg_sub]
    if Mach > 1:
        if const == 'nu': 
            return [cons_nu_sup*k_turb, cons_nu_err_avg_sup]
        elif const == 'eta': 
            return [cons_eta_sup, cons_eta_err_avg_sup]

#Mach
'''replace this shit'''
def mach(m):
    if m<1: 
        M = '0p'+str(int(10*m))
        p_val = 4/3
    else: 
        M = str(int(m))
        p_val = 3/2
    return M, p_val

def Mach_str(Mach):
    if Mach > 1: str(int(m)) 
    else: return '0p'+str(int(10*m))

def get_p_val(Mach):
    if Mach > 1: return 3/2 
    else: return 4/3

def get_params_ret(ret):
    get_ind = []; param_lst = []
    ID = list(ret.pnames); vals = list(ret.popt); errs = list(ret.perr)
    for i, val in enumerate(vals): param_lst.append([val]+[abs(errs[i][0]),errs[i][1]])
    params = dict(zip(ID, param_lst))
    return params

def get_avg_err_ret(lst):
    err = (abs(lst[1]) + lst[2])/2
    return lst[0], err

def perform_fit(model,x,y,yerr,param,log=True):

    def get_mix_err(err1,err2):
        low = np.sqrt(np.square(err1[0])+np.square(err2[0]))
        up = np.sqrt(np.square(err1[1])+np.square(err2[1]))
        return [low,up]

    if log: 
        try: yerr = yerr/(y*np.log(10))
        except: pass
        y = np.log10(y)
    
    ret_sys = cfp.fit(model, x, y, perr_method='systematic', params = param, dat_frac_for_systematic_perr=0.3)
    ret_stat = cfp.fit(model, x, y, yerr = yerr, perr_method='statistical', params = param)

    params = {}
    param_sys = get_params_ret(ret_sys)
    param_stat = get_params_ret(ret_stat)

    for name, val_lst in param_stat.items():
        params[name] = [val_lst[0],*get_mix_err(val_lst[1:],param_sys[name][1:])]

    return params

def text_1(arr1,arr2,n):
    if isinstance(arr1[0], str): arr1 = arr1[1:]; arr2 = arr2[1:] 
    ans = []
    for i in range(len(arr1)):
        if arr1[i]<1e-8: arr1[i] = 0
        val = cfp.round(arr1[i],nfigs=n)
        err = cfp.round(arr2[i],nfigs=2)
        ans.append(f'${val}'+r'\!\pm\!'+f'{err}$')
    return ans

def text_2(arr1,arr2,arr3):

    try: check = iter(arr1)
    except: arr1 = [arr1]; arr2 = [arr2]; arr3 = [arr3]

    ans = []
    for j in range(len(arr1)):
        i = 0
        while (arr1[j]>10):
            arr1[j] = arr1[j]/10
            i = i+1
        div = np.power(10,i)
        val = cfp.round(arr1[j],nfigs=2)
        err1 = cfp.round(arr2[j]/div,nfigs=1)
        err2 = cfp.round(arr3[j]/div,nfigs=1)
        if i!=0: ans.append(f'${val}'+r'_{-'+f'{err1}'+'}'+r'^{+'+f'{err2}'+'}'+r'\!\times\! 10^{'+f'{i}'+'}$')
        else: ans.append(f'${val}'+r'_{-'+f'{err1}'+'}'+r'^{+'+f'{err2}'+'}$')
    return ans

def text_3(arr1,arr2):
    ans = []
    for i in range(len(arr1)):
        if arr2[i]<1e-2: arr2[i] = '0.00'
        val = cfp.round(arr1[i],nfigs=3)
        err = cfp.round(arr2[i],nfigs=2)
        ans.append(f'${val}'+r'\!\pm\!'+f'{err}$')
    return np.array(ans)

def Label(name):
    if name == 'N_Re': return r'$N_{\mathrm{Re}} \!= \!'
    if name == 'p_Re': return r'$p_{\mathrm{Re}} \!= \!'
    if name == 'N_Rm': return r'$N_{\mathrm{Rm}} \!= \!'
    if name == 'p_Rm': return r'$p_{\mathrm{Rm}} \!= \!'
    if name == 'N_Pm': return r'$N_{\mathrm{Pm}} \!= \!'
    if name == 'p_Pm': return r'$p_{\mathrm{Pm}} \!= \!'
    if name == 'Pm_0': return r'$\mathrm{Pm}_0 \!= \!'
    if name == 'con': return r'$P_{\mathrm{const}} \!= \!'
    if name == 'cons_nu': return r'$c_{\nu} \!= \!'
    if name == 'cons_eta': return r'$c_{\eta} \!= \!'

error_text = 'N/A'

#independent fits
def plot_creat_ret(model, param, N_Res):
    
    args_ = list(inspect.getargspec(model)[0])
    if isinstance(param, dict): pass
    else: param = get_params_ret(param)
    
    legend = ''; val_dict = {}

    for name, val_lst in param.items():
        if name in args_: pass
        else: continue

        val, err = val_lst[0], val_lst[1:]
        val_dict[name] = val
        if len(err) > 1: asym = True
        else: asym = False; err = err[0]
        try: fix = all(v == 0. for v in err)
        except: fix = (err == 0.)

        if ('p_val' in name) or ('cons' in name): pass
        else:
            legend += Label(name) + f'{val:.2f}'
            if err and (not fix): 
                if asym: legend += r'_{-'+f'{err[0]:.2f}'+'}'+r'^{+'+f'{err[1]:.2f}'+'}'+'$'
                else: legend += r'\,\!\pm\,'+f'{err:.2f}'+'$'
            else: legend += r'\:\pm\:'+'$'+error_text
            legend += ', '
    
    if legend[-2:] == ', ': legend = legend[:-2]
    y = 10**model(N=N_Res, **val_dict)
    return y, legend

def plot_creat(model, fit, N_Res):
    error_text = 'N/A'
    legend = ''; Name = []; Values = []; i=0
    for name, param in fit.params.items():
        Name.append(name); Values.append(param.value)
        
    if ('p_val' not in Name):
        for name, param in fit.params.items():
            if param.stderr == 0 or param.stderr == None: legend = legend + Label(name) + f'{param.value:.2f}'+r'\:\pm\:'+'$'+error_text
            else: legend = legend + Label(name) + f'{param.value:.2f}'+r'\,\!\pm\,'+f'{param.stderr:.2f}'+'$'
            legend = legend + ', '
            i = i+1
    else:
        i = 0
        for name, param in fit.params.items():
            if (name != 'p_val') and (name != 'cons_nu') and (name != 'cons_eta'):
                if param.stderr == 0 or param.stderr == None: legend = legend + Label(name) + f'{param.value:.2f}'+r'\:\pm\:'+'$'+error_text
                else: legend = legend + Label(name) + f'{param.value:.2f}'+r'\,\!\pm\,'+f'{param.stderr:.2f}'+'$'
                legend = legend +', '
            i = i+1
    
    if legend[-2:] == ', ': legend = legend[:-2]
    params_dict = dict(zip(Name, Values))
    y = 10**model(N=N_Res, **params_dict)
    return y, legend

#dependent fits: taking params from prev fits


def plot_creat_mod(model, fit, error, N_res):
    legend = ''; Name = []; Values = []; i=0

    for name, param in fit.params.items():
        Name.append(name); Values.append(param.value)

    if ('p_val' not in Name):
        for name, param in fit.params.items():
            if param.value>100: 
                if error[i]==0 or error[i]==None: legend = legend + Label(name) + f'{int(param.value)}'+r'\:\pm\:'+'$'+error_text
                else: legend = legend + Label(name) + f'{int(param.value)}'+r'\,\!\pm\,'+f'{int(error[i])}'+'$'
            else:
                if error[i]==0 or error[i]==None: legend = legend + Label(name) + f'{param.value:.2f}'+r'\:\pm\:'+'$'+error_text
                else: legend = legend + Label(name) + f'{param.value:.2f}'+r'\,\!\pm\,'+f'{error[i]:.2f}'+'$'
            legend = legend + ', '
            i = i+1
    else:
        i = 0
        for name, param in fit.params.items():
            if (name != 'p_val') and (name != 'cons_nu') and (name != 'cons_eta'):
                if error[i]==0 or error[i]==None: legend = legend + Label(name) + f'{param.value:.2f}'+r'\:\pm\:'+'$'+error_text
                else: legend = legend + Label(name) + f'{param.value:.2f}'+r'\,\!\pm\,'+f'{error[i]:.2f}'+'$'
                legend = legend + ', '
                i = i+1

    if legend[-2:] == ', ': legend = legend[:-2]
    params_dict = dict(zip(Name, Values))
    y = 10**model(N=N_res, **params_dict)
    return y, legend

def add_errorbar(ax, y, y_err, N_res,marker='o',colr='k',label='Simulation'):
    line = ax.errorbar(N_res,y,yerr=2*y_err,ls='none',marker=marker,mfc='white',mec=colr,barsabove=True,elinewidth=0.85,ecolor=colr,capsize=3,label=label)
    return line
    
def add_params_theory(N_res, ax, model, p_val, values, color):
    N_theo = np.linspace(np.min(N_res),np.max(N_res),1000)

    arguments = inspect.getfullargspec(model).args
    p_val_exist = any('p_val' in arguments for item in arguments) 
    cons_nu_exist = any('cons_nu' in arguments for item in arguments) 
    cons_eta_exist = any('cons_eta' in arguments for item in arguments)
    if cons_nu_exist and cons_eta_exist: cons_end = 2
    elif cons_nu_exist or cons_eta_exist: cons_end = 1
    else: cons_end = 0

    if p_val_exist: 
        arguments = arguments[1:-1]
        value_dict = dict(zip(arguments, values))
        y = 10**model(N=N_theo,**value_dict,p_val = p_val)
    else: 
        arguments = arguments[1:]
        value_dict = dict(zip(arguments, values))
        y = 10**model(N=N_theo,**value_dict)

    legend = ''
    if cons_end: legend_args = arguments[:-cons_end]
    else: legend_args = arguments

    for i, args in enumerate(legend_args):
        legend = legend + Label(args) + f'{values[i]:.2f}'+r'\:\pm\:'+'$'+error_text
        if i != len(legend_args)-1: legend = legend +f', '
    ax.plot(N_theo,y,':',color=color,label=legend)
    return

def add_params_fit(model, p_val, N_res):
    params = Parameters()
    for argu in inspect.getfullargspec(model).args:
        if argu != 'N': params.add(argu)
        if argu == 'N_Re': 
            params[argu].min = 0
        if argu == 'N_Rm': 
            params[argu].value = 1.5
            params[argu].min = 0; params[argu].max = 4.0
        if argu == 'p_Re':
            params[argu].min = 0; params[argu].max = 2.0
        if argu == 'p_Rm':
            params[argu].value = p_val
            if p_val == 3/2: params[argu].value = 1.9
            params[argu].min = 0; params[argu].max = 3.0
        if argu == 'p_val': 
            params[argu].value = p_val
            params[argu].vary = False
    mod = Model(model)
    return params, mod

#Ng_error:
n_ran = int(1e4); bins = 1000

def ng_error(c, log = True):
    # Now get some statistics of c
    # first get the PDF data and define the binsize
    if log: ret = cfp.get_pdf(np.log10(c), range=None, bins=bins)
    else: ret = cfp.get_pdf(c, range=None, bins=bins)
    pdf = ret.pdf; bin_centre = ret.bin_center
    binsize = bin_centre[1]-bin_centre[0] # this is the bin width
    loc = bin_centre - 0.5*binsize # this is the left edge of each bin location
    cdf = np.cumsum(pdf)*binsize # this is the cumulative distribution function

    # now compute the 16th and 84th percentile
    index = cdf >= 0.16
    if log: sixteenth = 10**(loc[index][0])
    else: sixteenth = loc[index][0]
    index = cdf >= 0.84
    if log: eightyfourth = 10**(loc[index][0])
    else: eightyfourth = loc[index][0]
    # compute the median
    index = cdf >= 0.5
    if log: median_c = 10**(loc[index][0])
    else: median_c = loc[index][0]

    # now pick your preferred best value; here let's pick the median
    err_low = median_c - sixteenth
    err_high = eightyfourth - median_c
    
    return median_c, err_low, err_high
    
# ============================================== Models ============================================== 
#k_nu
def k_nu_of_Re(Re,cons_nu,p_val):
    p_val_inv = 1/p_val
    return cons_nu*Re**p_val_inv
    
def log_k_nu_of_N(N,N_Re,p_Re,cons_nu,p_val):
    Re = (N/N_Re)**p_Re
    return np.log10(k_nu_of_Re(Re,cons_nu,p_val))

#Kriel relation: k_eta(Rm)
def k_eta_of_Rm(Rm,Re,cons_nu,cons_eta,p_val):
    Pm = Rm/Re
    return cons_eta*k_nu_of_Re(Re,cons_nu,p_val)*Pm**0.5
    
#Fit equation: k_eta(N_res)
def log_k_eta_of_N(N,N_Re,p_Re,N_Rm,p_Rm,cons_nu,cons_eta,p_val):
    Re = (N/N_Re)**p_Re
    Rm = (N/N_Rm)**p_Rm
    return np.log10(k_eta_of_Rm(Rm,Re,cons_nu,cons_eta,p_val))

def log_k_eta_lin(N,N_prop):
    return np.log10(N_prop*N)

#Kriel relation: Re(k_nu)
def Re_of_k_nu(k_nu,cons_nu,p_val):
    return (k_nu/cons_nu)**p_val

#Fit equation: Re(N_res)
def log_Re_of_N(N,N_Re,p_Re):
    return np.log10((N/N_Re)**p_Re)

#Kriel relation: Pm(k_eta,k_nu)
def Pm_of_k_eta(k_eta,k_nu,cons_eta):
    return (k_eta/(cons_eta*k_nu))**2

#Fit equation: Pm(N_res)
def log_Pm_of_N(N,N_Re,p_Re,N_Rm,p_Rm):
    Re = (N/N_Re)**p_Re
    Rm = (N/N_Rm)**p_Rm
    Pm = Rm/Re
    return np.log10(Pm)

def pPm(p_Rm,p_Re):
    p_Pm = p_Rm - p_Re
    return p_Pm

def Pm0(N_Re,N_Rm,p_Re,p_Rm):
    Pm_0 = (N_Re**p_Re)/(N_Rm**p_Rm)
    return Pm_0
    
def log_Pm_form(N,Pm_0,p_Pm):
    Pm = Pm_0*(N**p_Pm)
    return np.log10(Pm)

def Pm_of_N_log(N,con):
    log_Pm = con + 0*N
    return log_Pm

#Kriel relation: Rm(k_eta); using other 2 relations
def Rm_of_k_eta(Pm,Re):
    return Pm*Re

#Fit equation: Rm(N_res)
def log_Rm_of_N(N,N_Rm,p_Rm):
    Rm = (N/N_Rm)**p_Rm
    return np.log10(Rm)

# ============================================== Errors ============================================== 
def Re_ng_error(K_nu,K_nu_err,cons_nu,cons_nu_err_avg,p_val):

    try: check = iter(K_nu)
    except: K_nu = [K_nu]; K_nu_err = [K_nu_err]

    Re_Ran = []; Re_Med = []; Re_Err_low = []; Re_Err_high = []
    Norm = cfp.generate_random_gaussian_numbers(n=2*n_ran, mu=0, sigma=1, seed=111)
    leng = len(K_nu)

    for i in range(leng):
        k_nu = K_nu[i]; k_nu_err = K_nu_err[i]
        k_nu_ran = k_nu + k_nu_err*Norm[:n_ran]
        cons_nu_ran = cons_nu + cons_nu_err_avg*Norm[n_ran:]
        Re_ran = k_nu_ran/cons_nu_ran
        Re_ran = Re_ran[Re_ran>=0]**p_val
        Re_med, Re_err_low, Re_err_high = ng_error(Re_ran, log = True)
        if leng == 1: return Re_med, Re_err_low, Re_err_high, Re_ran

        Re_Med.append(Re_med); Re_Err_low.append(Re_err_low); Re_Err_high.append(Re_err_high); Re_Ran.append(Re_ran)

    return np.array(Re_Med), np.array(Re_Err_low), np.array(Re_Err_high), np.array(Re_Ran, dtype=object)

def pPm_err(p_Re,p_Rm,p_Re_err,p_Rm_err):
    Norm = cfp.generate_random_gaussian_numbers(n=2*n_ran, mu=0, sigma=1, seed=111)
    p_Re_ran = p_Re + p_Re_err*Norm[:n_ran]
    p_Rm_ran = p_Rm + p_Rm_err*Norm[n_ran:]
    pPm_ran = pPm(p_Rm_ran,p_Re_ran)
    return ng_error(pPm_ran, log = False)

def Pm0_err(N_Re,N_Rm,p_Re,p_Rm,N_Re_err,N_Rm_err,p_Re_err,p_Rm_err):
    Norm = cfp.generate_random_gaussian_numbers(n=4*n_ran, mu=0, sigma=1, seed=111)
    N_Re_ran = N_Re + N_Re_err*Norm[:n_ran]
    N_Rm_ran = N_Rm + N_Rm_err*Norm[n_ran:2*n_ran]
    #N_Rm_ran = 10**(np.log10(N_Rm) + np.log10(1+N_Rm_err/N_Rm)*Norm[n_ran:2*n_ran])
    #N_Rm_ran = N_Rm + N_Rm_ran - N_Rm_ran.mean()
    p_Re_ran = p_Re + p_Re_err*Norm[2*n_ran:3*n_ran]
    p_Rm_ran = p_Rm + p_Rm_err*Norm[3*n_ran:]

    N_Re_ran = N_Re_ran[N_Re_ran>=0]
    p_Re_ran = p_Re_ran[:len(N_Re_ran)]
    arr = N_Re_ran**p_Re_ran

    N_Rm_ran = N_Rm_ran[N_Rm_ran>=0]
    p_Rm_ran = p_Rm_ran[:len(N_Rm_ran)]
    array = N_Rm_ran**p_Rm_ran

    len_ = min(len(array),len(arr))
    arr = arr[:len_]
    array = array[:len_]
    Pm0_ran = arr/array

    return ng_error(Pm0_ran, log = True)

def Pm_ng_error(K_nu,K_nu_err,K_eta,K_eta_err,cons_eta,cons_eta_err_avg,p_val):

    try: check = iter(K_nu)
    except: K_nu = [K_nu]; K_nu_err = [K_nu_err]; K_eta = [K_eta]; K_eta_err = [K_eta_err]
    leng = len(K_nu)

    Pm_Ran = []; Pm_Med = []; Pm_Err_low = []; Pm_Err_high = []
    Norm = cfp.generate_random_gaussian_numbers(n=3*n_ran, mu=0, sigma=1, seed=111)

    for i in range(len(K_nu)):
        k_nu = K_nu[i]; k_nu_err = K_nu_err[i]; k_eta = K_eta[i]; k_eta_err = K_eta_err[i]
        k_nu_ran = k_nu + k_nu_err*Norm[:n_ran]
        k_eta_ran = k_eta + k_eta_err*Norm[n_ran:2*n_ran]
        cons_eta_ran = cons_eta + cons_eta_err_avg*Norm[2*n_ran:]
        Pm_ran = (k_eta_ran/(cons_eta_ran*k_nu_ran))**2
        Pm_med, Pm_err_low, Pm_err_high = ng_error(Pm_ran, log = True)

        if leng == 1: return Pm_med, Pm_err_low, Pm_err_high, Pm_ran
        Pm_Med.append(Pm_med); Pm_Err_low.append(Pm_err_low); Pm_Err_high.append(Pm_err_high); Pm_Ran.append(Pm_ran)

    return np.array(Pm_Med), np.array(Pm_Err_low), np.array(Pm_Err_high), np.array(Pm_Ran)

def Rm_ng_error(Rm_Ran): 
    shap = np.shape(Rm_Ran)
    if len(shap) > 1:
        Rm_Med = []; Rm_Err_low = []; Rm_Err_high = []
        for i in range(len(Rm_Ran)):
            Rm_ran = Rm_Ran[i]
            Rm_med, Rm_err_low, Rm_err_high = ng_error(Rm_ran, log = True)
            Rm_Med.append(Rm_med); Rm_Err_low.append(Rm_err_low); Rm_Err_high.append(Rm_err_high)

    else: return ng_error(Rm_Ran, log = True)
    return np.array(Rm_Med), np.array(Rm_Err_low), np.array(Rm_Err_high)

def get_col(file,col,n):
    col_val = (line.split()[col-1] for line in file[n:])
    col_val = [*col_val]
    if all(c.isalpha() or c=='_' for c in col_val[0]):
        col_val = [col_val[0]] + list(map(float,col_val[1:]))
    else: col_val = list(map(float,col_val))
    return col_val

def make_tab(Params):
    Tab_list = []; Ind = list(range(len(Params)))
    while len(Ind):
        i = Ind[0]; param = Params[i]
        if i != Ind[-1]:
            if 'err' in Params[i+1][0]: 
                if min(param[1:]) < 1: round_num = 2
                else: round_num = 3
                Tab_list.append([param[0]] + text_1(param[1:],Params[i+1][1:],round_num))
                Ind.pop(1)
            else: Tab_list.append(param)
            Ind.pop(0)
        else: 
            Tab_list.append(Params[-1])
            Ind.pop()
    return Tab_list

#****************************************************************************************************
def Index(k,N):
    ind = ((k>=3) & (k<=N/4))
    return ind

def P_kin_fit_eq(k,A,p_kin,p_bn,k_bn,k_nu,p_nu):
    k_scale = k/k_bn
    P = A*(k_scale**p_kin + k_scale**p_bn)*np.exp(-(k/k_nu)**p_nu)
    return P

def P_mag_fit_eq(k,A,p_mag,k_eta,p_eta):
    Bessel = sp.kn(0,(k/k_eta)**p_eta)
    if np.any(np.isnan(Bessel)): stop()
    P = A*k**p_mag*Bessel
    if np.any(np.isnan(P)) or np.any(np.isinf(np.abs(P))): stop()
    return P