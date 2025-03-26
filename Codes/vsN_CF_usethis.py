import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from cfpack import matplotlibrc
import cfpack as cfp
from cfpack import stop, print
import random
import numpy as np
from lmfit import Model, Parameters
import scipy.special as sp
import inspect
import textwrap
from vsN_funcs import *
from fit import myfit
np.seterr(divide='ignore', invalid='ignore')
home_directory = os.path.expanduser( '~' )+'/'

textname = []
M_str = []; #stores the string value of Mach no.
P_val = []; #stores the Kriel value of wavenumber and Reynold's no. dependence
#Stores wavenumbers obtained from power spectra and their errors
K_nu = []; K_nu_err = []; K_eta = []; K_eta_err = []
#stores parameters from k vs. N fits when all are left free
Params_bfre = {}; Params_nfre = {}
#stores parameters from k vs. N fits when N_Re/m is left free and p_Re/m is fixed
N_Re_nref = []; p_Re_nref = []; N_Re_nref_err = []; p_Re_nref_err = []
N_Re_pref = []; p_Re_pref = []; N_Re_pref_err = []; p_Re_pref_err = []
N_Rm_nref = []; p_Rm_nref = []; N_Rm_nref_err = []; p_Rm_nref_err = []
N_Rm_bfre = []; p_Rm_bfre = []; N_Rm_bfre_err = []; p_Rm_bfre_err = [] 

Mach = [0.1,10]
#Mach = [10]
Const = ['nu','eta']
Spect = ['kin','mag']

#File access
folder = 'Simu_results'
target = home_directory+'Num_diss_results/'
def file_access(M,spect,const,n):
    textname = 'k_'+const+'_vs_N_'+M
    Path = home_directory+folder+'/'+textname+'.txt'
    file = open(Path).readlines()
    get_col = lambda col: (line.split()[col-1] for line in file[5:n])
    return textname,get_col

#All the necessary data from fit of spectra is extracted
for spect in Spect:
    for m in Mach:
        M, p_val = mach(m)
        M_str.append(M)
        P_val.append(p_val)
        if spect == 'kin':
            const = 'nu'
            Textname,get_col = file_access(M,spect,const,11)
            N_res = np.array(list(map(float,get_col(1)))) 
            k_nu = np.array(list(map(float,get_col(10))))
            k_nu_err = np.array(list(map(float,get_col(11))))
            textname.append(Textname)
            K_nu.append(k_nu)
            K_nu_err.append(k_nu_err)
        
        if spect == 'mag':
            const = 'eta'
            Textname,get_col = file_access(M,spect,const,11)
            N_res = np.array(list(map(float,get_col(1))))
            k_eta = np.array(list(map(float,get_col(8))))
            k_eta_err = np.array(list(map(float,get_col(9))))
            textname.append(Textname)
            K_eta.append(k_eta)
            K_eta_err.append(k_eta_err)
textname = np.array(textname)

#Define:
N_theo = np.linspace(np.min(N_res),np.max(N_res),1000)
elinewidth=0.85; ecolor='k'; capsize=3

# ============================================== K_NU & K_ETA ===============================================
#PLOTS:
filename = 'k_vs_N.pdf'
fig = plt.figure(figsize = (12,6))
plt.rcParams.update({'font.size': 12})
fontsize = 8.1; location = 'lower right'; markerfirst=False; align = 'right'

ax1 = plt.subplot(221)
ax2 = plt.subplot(222, sharey = ax1)
ax3 = plt.subplot(223, sharex = ax1)
ax4 = plt.subplot(224, sharex = ax2, sharey = ax3)

#sharing x
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
#sharing y
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax4.get_yticklabels(), visible=False)
plt.subplots_adjust(wspace=0.02, hspace=0.05)

# ==============================================  K_NU ============================================== 
#Parameters:
Ax = [ax1,ax2]; model = log_k_nu_of_N; order = [5,0,1,3,4,2]; n_re_mach = [1.5,3]
for i in range(len(Mach)):
    k_nu = K_nu[i]; k_nu_err = K_nu_err[i]; p_val = get_p_val(Mach[i])
    ax = Ax[i]; weights = k_nu/(2*k_nu_err); cons_nu = cons(Mach[i],'nu')[0]

    #Data & Errorbar:
    add_errorbar(ax = ax, y = k_nu, y_err = k_nu_err, N_res = N_res)

    #Theory
    params = [1.0,p_val,cons_nu]
    add_params_theory(N_res = N_res, ax = ax, model = model, p_val = p_val, values = params, color = 'k')
    params = [n_re_mach[i],p_val,cons_nu]
    add_params_theory(N_res = N_res, ax = ax, model = model, p_val = p_val, values = params, color = 'g')
    
    #All free params
    #print(' ********************* ALL FREE ********************** ')
    my_params = {
        'N_Re': [n_re_mach[i]*0.5, n_re_mach[i], n_re_mach[i]*5],
        'p_Re': [p_val*0.8, p_val, p_val*2],
        'cons_nu': [None, cons_nu, None],
        'p_val': [None, p_val, None]
    }
    
    params_bfre = perform_fit(model, N_res, k_nu, k_nu_err, my_params)
    Params_bfre[Mach[i]] = params_bfre
    
    y, label = plot_creat_ret(model, params_bfre, N_res)
    ax.plot(N_res,y,'-',color='orange',label= label)

    #Saving data:
    with open(folder+'/'+textname[i]+'.txt', 'r') as file: data = file.readlines()
    data[0] = f"N_Re = {params_bfre['N_Re'][0]}\n"
    data[1] = f"N_Re_error = {params_bfre['N_Re'][1]}+{params_bfre['N_Re'][2]}\n"
    data[2] = f"p_Re = {params_bfre['p_Re'][0]}\n"
    data[3] = f"p_Re_error = {params_bfre['p_Re'][1]}+{params_bfre['p_Re'][2]}\n"
    with open(folder+'/'+textname[i]+'.txt', 'w') as file: file.writelines(data)
    #print(f'Saved {textname[i]}')

    #print(' ********************* p_Re FREE ********************* ')
    my_params['N_Re'] = [None, n_re_mach[i], None]
    y, label = plot_creat_ret(model, perform_fit(model, N_res, k_nu, k_nu_err, my_params), N_res)
    ax.plot(N_res,y,'--',color='magenta',label=label)

    #print(' ********************* N_Re FREE ********************* ')
    my_params['N_Re'] = [n_re_mach[i]*0.5, n_re_mach[i], n_re_mach[i]*10]
    my_params['p_Re'] = [None, p_val, None]

    params_nfre = perform_fit(model, N_res, k_nu, k_nu_err, my_params)
    Params_nfre[Mach[i]] = params_nfre
    y, label = plot_creat_ret(model, params_nfre, N_res)
    ax.plot(N_res,y,'--',color='blue',label=label)

    #Design:
    ax.set_xscale('log')
    ax.set_yscale('log')
    if ax == Ax[0]: ax.set_ylabel(r'$k_{\nu}$')
    handles, labels = fig.get_axes()[i].get_legend_handles_labels()
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=fontsize, loc=location, markerfirst=markerfirst)

# ============================================== K_ETA ===============================================
#Parameters:
Ax = [ax3,ax4]; model = log_k_eta_of_N; order = [3,0,1,2]
for i in range(len(Mach)):
    k_eta = K_eta[i]; k_eta_err = K_eta_err[i]; p_val = get_p_val(Mach[i])
    ax = Ax[i]; weights = k_eta/(2*k_eta_err); m = Mach[i]
    cons_nu = cons(Mach[i],'nu')[0]; cons_eta = cons(Mach[i],'eta')[0]

    #Data & Errorbar:
    add_errorbar(ax = ax, y = k_eta, y_err = k_eta_err, N_res = N_res)

    #Theory (black dot line)
    params = [n_re_mach[i],p_val,n_re_mach[i],p_val,cons_nu,cons_eta]
    add_params_theory(N_res = N_res, ax = ax, model = model, p_val = p_val, values = params, color = 'k')

    #Fitting:
    # *************************** N_Re free params ***************************
    params_nfre = Params_nfre[m]
    my_params = {
        'N_Re': [None, params_nfre['N_Re'][0], None],
        'p_Re': [None, params_nfre['p_Re'][0], None],
        'N_Rm': [n_re_mach[i]*0.1, n_re_mach[i], n_re_mach[i]*5],
        'p_Rm': [None, p_val, None],
        'cons_nu': [None, cons_nu, None],
        'cons_eta': [None, cons_eta, None],
        'p_val': [None, p_val, None]
    }

    params_ = perform_fit(model, N_res, k_eta, k_eta_err, my_params)
    params_nfre['N_Rm'] = params_['N_Rm']
    params_nfre['p_Rm'] = params_['p_Rm']
    params_nfre['cons_eta'] = params_['cons_eta']
    Params_nfre[m] = params_nfre
    
    y, label = plot_creat_ret(model, params_nfre, N_res)
    ax.plot(N_res,y,'--',color='blue',label=label)

    # *************************** both free params ***************************
    params_bfre = Params_bfre[m]
    my_params['N_Re'][1] = params_bfre['N_Re'][0]
    my_params['p_Re'][1] = params_bfre['p_Re'][0]
    my_params['p_Rm'] = [p_val*0.1, p_val, p_val*2]
    
    params_ = perform_fit(model, N_res, k_eta, k_eta_err, my_params)
    params_bfre['N_Rm'] = params_['N_Rm']
    params_bfre['p_Rm'] = params_['p_Rm']
    params_bfre['cons_eta'] = params_['cons_eta']
    Params_bfre[m] = params_bfre

    #Saving data:
    with open(folder+'/'+textname[i]+'.txt', 'r') as file: data = file.readlines()
    data[0] = f"N_Rm = {params_bfre['N_Rm'][0]}\n"
    data[1] = f"N_Rm_error = {params_bfre['N_Rm'][1]}+{params_bfre['N_Rm'][2]}\n"
    data[2] = f"p_Rm = {params_bfre['p_Rm'][0]}\n"
    data[3] = f"p_Rm_error = {params_bfre['p_Rm'][1]}+{params_bfre['p_Rm'][2]}\n"
    with open(folder+'/'+textname[i]+'.txt', 'w') as file: file.writelines(data)

    y, label = plot_creat_ret(model, params_bfre, N_res)
    ax.plot(N_res,y,'-',color='orange',label=label)

    # *************************** Linear fit k_eta ***************************

    if m < 1: N_prop_guess = 0.03
    if m > 1: N_prop_guess = 0.07
    my_params = {'N_prop': [N_prop_guess*0.1, N_prop_guess, N_prop_guess*10]}
    params_ = perform_fit(log_k_eta_lin, N_res, k_eta, k_eta_err, my_params)
    N_prop, N_prop_err_low, N_prop_err_up = params_['N_prop']

    y = 10**log_k_eta_lin(N = N_res, N_prop = N_prop)
    ax.plot(N_res,y,'-.',color='red')
    ax.axhline(y=9e1,xmin=0.025,xmax=0.125,linestyle = '-.',color='red')
    
    lin_rel = r'$k_{\eta} = '+f'({N_prop:.3f}'+ r'_{-'+f'{N_prop_err_low:.3f}'+'}'+r'^{+'+f'{N_prop_err_up:.3f}'+'}'+')N$'
    ax.text(0.3, 0.86, lin_rel, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontsize=11)

    #Design:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([0.2,210])
    ax.set_xlim([np.min(N_res)-20,np.max(N_res)+500])
    if ax == Ax[0]: ax.set_ylabel(r'$k_{\eta}$')
    ax.set_xlabel('$N$')
    handles, labels = fig.get_axes()[i+2].get_legend_handles_labels()
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=fontsize, loc=location, markerfirst=markerfirst)

#Mach label
M = r'$\mathcal{M}$'
ax1.text(0.11, 0.86, M+' = 0.1', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,fontsize=11,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))
ax2.text(0.11, 0.86, M+' = 10', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes,fontsize=11,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))

fig.savefig(target+filename,format = "pdf",bbox_inches='tight')
print(f'Saved {filename}')
plt.close()


#============================================== RE ===============================================
#Plotting:
filename = 'Rey_vs_N.pdf'
fig = plt.figure(figsize = (15,10))
plt.rcParams.update({'font.size': 15})
ax1 = plt.subplot(321)
ax2 = plt.subplot(322, sharey = ax1)
ax3 = plt.subplot(323, sharex = ax1)
ax4 = plt.subplot(324, sharey = ax3, sharex = ax2)
ax5 = plt.subplot(325, sharex = ax3)
ax6 = plt.subplot(326, sharey = ax5, sharex = ax4)
fontsize = 10.5; location = 'lower right'
#sharing x
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax4.get_xticklabels(), visible=False)
#sharing y
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax4.get_yticklabels(), visible=False)
plt.setp(ax6.get_yticklabels(), visible=False)
plt.subplots_adjust(wspace=0.02, hspace=0.05)

Re_value = []; Re_Ran = []; Re_Err_down = []; Re_Err_up = []; Re_Med = []; order = [3,0,1,2]
#Parameters:
Ax = [ax1,ax2]; model = log_Re_of_N
for i in range(len(Mach)):
    k_nu = K_nu[i]; k_nu_err = K_nu_err[i]; p_val = get_p_val(Mach[i]); m = Mach[i]
    ax = Ax[i]; cons_nu = cons(m,'nu')[0]

    #Calculating Re from data
    Re = Re_of_k_nu(k_nu,cons_nu,p_val); Re_value.append(Re)

    Re_med, Re_err_down, Re_err_up, Re_ran = Re_ng_error(k_nu,k_nu_err,*cons(m,'nu'),p_val)
    Re_Med.append(Re_med); Re_Ran.append(Re_ran)
    Re_Err_down.append(Re_err_down); Re_Err_up.append(Re_err_up)
    Re_asymm_err = np.array(list(zip(Re_err_down, Re_err_up))).T
    Re_err_avg = (Re_err_up+Re_err_down)/2

    #Data & Errorbar:
    weights=Re/(2*Re_err_avg)
    add_errorbar(ax = ax, y = Re_med, y_err = Re_asymm_err, N_res = N_res)

    #Theory
    params = [n_re_mach[i],p_val]
    add_params_theory(N_res = N_res,ax = ax, model = model, p_val = p_val, values = params, color = 'k')

    #print(' ********************* N_Re free ********************** ')
    params_nfre = Params_nfre[m]
    y, label = plot_creat_ret(model, params_nfre, N_res)
    ax.plot(N_res,y,'--',color='blue',label=label)

    #print(' ********************* Both left free ********************** ')
    params_bfre = Params_bfre[m]
    y, label = plot_creat_ret(model, params_bfre, N_res)
    ax.plot(N_res,y,'-',color='orange',label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([80,3e5])
    if ax == Ax[0]: ax.set_ylabel(r'Re')
    handles, labels = fig.get_axes()[i].get_legend_handles_labels()
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=fontsize, loc=location, markerfirst=markerfirst)

# ============================================== PM ===============================================

Pm_value = []; Pm_Ran = []; Pm_Med = []; Pm_Err_down = []; Pm_Err_up = []
#Parameters:
Ax = [ax3,ax4]
for i in range(len(Mach)):
    k_nu = K_nu[i]; k_nu_err = K_nu_err[i]
    k_eta = K_eta[i]; k_eta_err = K_eta_err[i]
    ax = Ax[i]; p_val = get_p_val(Mach[i]); m = Mach[i]
    cons_eta = cons(Mach[i],'eta')[0]

    #Calculating Pm
    Pm = Pm_of_k_eta(k_eta,k_nu,cons_eta); Pm_value.append(Pm)
    Pm_med, Pm_err_down, Pm_err_up, Pm_ran = Pm_ng_error(k_nu,k_nu_err,k_eta,k_eta_err,*cons(m,'eta'),p_val)
    Pm_Med.append(Pm_med); Pm_Ran.append(Pm_ran)
    Pm_Err_down.append(Pm_err_down); Pm_Err_up.append(Pm_err_up)
    Pm_asymm_err = np.array(list(zip(Pm_err_down, Pm_err_up))).T
    Pm_err_avg = (Pm_err_up+Pm_err_down)/2

    #Data & Errorbar:
    weights = Pm/(2*Pm_err_avg)
    add_errorbar(ax = ax, y = Pm_med, y_err = Pm_asymm_err, N_res = N_res)

    #Fitting:
    model = Pm_of_N_log
    if m < 1: con_guess = 0.1
    if m > 1: con_guess = 0.7
    my_params = {'con': [con_guess*0.1, con_guess, con_guess*10]}

    params_ = perform_fit(model, N_res, Pm, Pm_err_avg, my_params)
    con, con_err_low, con_err_up = params_['con']

    y = 10**model(N = N_res, con = con)
    ax.axhline(y=10**con,xmin=0.045,xmax=0.96,linestyle = '-.',color='red')
    ax.axhline(y=22.5,xmin=0.025,xmax=0.10,linestyle = '-.',color='red')
    lin_rel = r'Pm = '+'$'+f'{10**con:.1f}'+ r'_{-'+f'{np.log(10)*10**con*con_err_low:.1f}'+'}'+r'^{+'+f'{np.log(10)*10**con*con_err_up:.1f}'+'}'+'$'
    ax.text(0.2, 0.93, lin_rel, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontsize=12)
    

    model = log_Pm_form
    # ********************************* Pm = 1 *********************************
    p_Pm = pPm(p_Rm=p_val,p_Re=p_val)
    Pm_0 = Pm0(N_Re=1.50,N_Rm=1.50,p_Re=p_val,p_Rm=p_val)
    add_params_theory(N_res = N_res,ax = ax, model = model, p_val = p_val, values = [Pm_0,p_Pm], color = 'k')

    # ********************************* N_re free *********************************
    params_nfre = Params_nfre[m]
    N_Re, N_Re_err = get_avg_err_ret(params_nfre['N_Re'])
    p_Re, p_Re_err = get_avg_err_ret(params_nfre['p_Re'])
    N_Rm, N_Rm_err = get_avg_err_ret(params_nfre['N_Rm'])
    p_Rm, p_Rm_err = get_avg_err_ret(params_nfre['p_Rm'])

    params_ = {
        'Pm_0': Pm0_err(N_Re, N_Rm, p_Re, p_Rm, N_Re_err, N_Rm_err, p_Re_err, p_Rm_err),
        'p_Pm': pPm_err(p_Re, p_Rm, p_Re_err, p_Rm_err)
    }
    y, label = plot_creat_ret(model, params_, N_res)
    ax.plot(N_res,y,'--',color='blue',label=label)
    
    # ********************************* N_re p_Re free *********************************

    params_bfre = Params_bfre[m]
    N_Re, N_Re_err = get_avg_err_ret(params_bfre['N_Re'])
    p_Re, p_Re_err = get_avg_err_ret(params_bfre['p_Re'])
    N_Rm, N_Rm_err = get_avg_err_ret(params_bfre['N_Rm'])
    p_Rm, p_Rm_err = get_avg_err_ret(params_bfre['p_Rm'])
    
    params_ = {
        'Pm_0': Pm0_err(N_Re, N_Rm, p_Re, p_Rm, N_Re_err, N_Rm_err, p_Re_err, p_Rm_err),
        'p_Pm': pPm_err(p_Re, p_Rm, p_Re_err, p_Rm_err)
    }
    y, label = plot_creat_ret(model, params_, N_res)
    ax.plot(N_res,y,'-',color='orange',label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$N$')
    ax.set_ylim([1e-2,3.5e1])
    if ax == Ax[0]: ax.set_ylabel(r'Pm')
    handles, labels = fig.get_axes()[i+2].get_legend_handles_labels()
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=fontsize, loc=location, markerfirst=markerfirst)

# ============================================== RM ===============================================
#Parameters:
Ax = [ax5,ax6]; model = log_Rm_of_N; Rm_Med = []; Rm_Err_down = []; Rm_Err_up = []
for i in range(len(Mach)):
    Re = Re_Med[i]; Pm = Pm_Med[i]
    ax = Ax[i]; p_val = get_p_val(Mach[i]); m = Mach[i]

    Rm = Rm_of_k_eta(Pm,Re)
    Rm_ran = (Pm_Ran[i]*Re_Ran[i]).astype(float)
    Rm_med, Rm_err_down, Rm_err_up = Rm_ng_error(Rm_ran)
    Rm_Med.append(Rm_med); Rm_Err_down.append(Rm_err_down); Rm_Err_up.append(Rm_err_up)
    Rm_asymm_err = np.array(list(zip(Rm_err_down, Rm_err_up))).T

    add_errorbar(ax = ax, y = Rm_med, y_err = Rm_asymm_err, N_res = N_res)
    params = [n_re_mach[i],p_val]
    add_params_theory(N_res = N_res,ax = ax, model = model, p_val = p_val, values = params, color = 'k')

    #Fitting:
    # ********************* N_Rm & p_Rm free ********************** 
    params_nfre = Params_nfre[m]
    y, label = plot_creat_ret(model, params_nfre, N_res)
    ax.plot(N_res,y,'--',color='blue',label=label)

    #print(' ********************* Both left free ********************** ')
    params_bfre = Params_bfre[m]
    y, label = plot_creat_ret(model, params_bfre, N_res)
    ax.plot(N_res,y,'-',color='orange',label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    if ax == Ax[0]: ax.set_ylabel(r'Rm')
    ax.set_xlabel('$N$')
    ax.set_ylim([1e1,5e5])
    ax.set_xlim([np.min(N_res)-20,np.max(N_res)+500])
    handles, labels = fig.get_axes()[i+4].get_legend_handles_labels()
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=fontsize, loc=location, markerfirst=markerfirst)
    
#Mach label
M = r'$\mathcal{M}$'
ax1.text(0.1, 0.86, M+' = 0.1', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,fontsize=15,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))
ax2.text(0.1, 0.86, M+' = 10', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes,fontsize=15,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))

fig.savefig(target+filename,format = "pdf",bbox_inches='tight')
print(f'Saved {filename}')
plt.close()

#********************************************  MERGE  **********************************************
def add_errorbar(ax, y, y_err, N_res):
    line = ax.errorbar(N_res,y,yerr=2*y_err,ls='none',marker='o',mfc='white',mec='k',barsabove=True,elinewidth=0.85,ecolor='k',capsize=3,label='This work: simulation')
    return line

#Plotting:
filename = 'Rey_vs_N_merge.pdf'
fig = plt.figure(figsize = (15,10))
plt.rcParams.update({'font.size': 15})
ax1 = plt.subplot(321)
ax2 = plt.subplot(322, sharey = ax1)
ax3 = plt.subplot(323, sharex = ax1)
ax4 = plt.subplot(324, sharey = ax3, sharex = ax2)
ax5 = plt.subplot(325, sharex = ax3)
ax6 = plt.subplot(326, sharey = ax5, sharex = ax4)
fontsize = 10; location = 'lower right'
#sharing x
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax4.get_xticklabels(), visible=False)
#sharing y
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax4.get_yticklabels(), visible=False)
plt.setp(ax6.get_yticklabels(), visible=False)
plt.subplots_adjust(wspace=0.02, hspace=0.05)

#[N,Re,Rm,Pm]
def lit_mod(arr):
    for i in range(len(arr)):
        arr_mod = arr[i]
        arr_mod[1] = arr_mod[1]*2*np.pi
        if arr_mod[2] != None:
            arr_mod[2] = arr_mod[2]*2*np.pi
    return arr

def lit_m():
    Lit = []; Label = []
    
    Sch4b = [[64,52,52,1],[128,100,100,1],[128,210,210,1],[256,450,450,1],[256,100,1000,10]]
    Sch4b = lit_mod(Sch4b); Lit.append(Sch4b); Label.append('SCT+(04)')

    Hau4 = [[128,280,280,1],[256,370,370,1],[512,930,930,1],[1024,1000,1000,1]]
    Hau4 = lit_mod(Hau4); Lit.append(Hau4); Label.append('HB(04)')

    Hau4a = [[64,120,120,1],[128,190,190,1],[256,420,420,1],[512,600,600,1],[1024,960,960,1]]
    Hau4a = lit_mod(Hau4a); Lit.append(Hau4a); Label.append('HBD(04)')

    Hau4M = [[256,43,43,1],[512,78,78,1]]    
    Hau4M = lit_mod(Hau4M); Lit.append(Hau4M); Label.append('HBM(04)')

    Brad06 = [[512,4,None,None],[512,50,None,None]]
    Brad06 = lit_mod(Brad06); Lit.append(Brad06); Label.append('MB(06)')

    Sch07 = [[64,51,51,1],[128,107,107,1],[128,210,210,1],[128,133,57,0.43],[128,460,57,0.125],[128,1000,109,0.109],[128,1100,230,0.21],[256,440,440,1],[256,440,110,0.25],[256,830,104,0.125],[256,440,220,0.5],[256,1760,220,0.125],[256,1760,310,0.176],[256,4300,470,0.11],[512,810,810,1],[512,5900,850,0.145]]
    Lit.append(Sch07); Label.append('SIC+(07)')

    Chris11 = [[128,1500,3000,2]]
    Lit.append(Chris11); Label.append('FCS+(11)')

    Brad19 = [[576,6800,1400,0.2],[576,960,6800,5]]
    Brad19 = lit_mod(Brad19); Lit.append(Brad19); Label.append('BR(19)')

    Rad21 = [[128,1500,3000,2]]
    Lit.append(Rad21); Label.append('AFT+(21)')

    Amit21 = [[512,2000,2000,1],[512,2000,6000,3]]
    Lit.append(Amit21); Label.append('SF(21)')
    
    Neco22 = [[288,430,430,1],[576,470,470,2],[288,3600,3600,1],[576,1700,1700,2]]
    Lit.append(Neco22); Label.append('KBS+(22)')

    Gal22 = [[280,920,920,1],[280,76,760,10],[560,2200,2200,1],[560,189,1890,10],[1120,10400,10400,1],[1120,940,9400,10],[2240,50000,50000,1],[2240,4800,48000,10]]
    Lit.append(Gal22); Label.append('GKW+(22)')

    Jam23 = [[288,500,500,1],[288,500,1000,2],[288,500,2000,4]]
    Lit.append(Jam23); Label.append('BFK+(23)')

    Gret23 = [[128,717,752,1],[256,1546,1943,1.3],[512,3389,4983,1.5],[1024,8030,12147,1.5],[2048,18571,27677,1.5]]
    Lit.append(Gret23); Label.append('GOB(23)')

    return Lit, Label

Lit, Lit_label = lit_m()
s = 20; Leg_han_m = []; Re_alt = []; N_alt = []; Rm_alt = []; Pm_alt = []; ax_1_leg = []
marker_type = ['o','^','1','s','>','+','*','p','d',7,'H','x','$O$','o']
color = ['red','blue','lime','orange','purple','gold','green','gray','brown','turquoise','teal','magenta','olive','indigo']
for i in range(len(Lit)):
    Data = Lit[i]; N_lit = []; Re_lit = []; Rm_lit = []; Pm_lit = [] 
    label = Lit_label[i]; mark_type = marker_type[i]; c=color[i]
    for i in range(len(Data)):
        Dat = Data[i]; N_lit.append(Dat[0]); Re_lit.append(Dat[1])
        if Dat[2] != None: Rm_lit.append(Dat[2]); Pm_lit.append(Dat[3])
        if label == 'GOB(23)': 
            N_alt.append(Dat[0]); Re_alt.append(Dat[1])
            Rm_alt.append(Dat[2]); Pm_alt.append(Dat[3])
    if label != 'GOB(23)':
        line = ax1.scatter(N_lit,Re_lit,s=s,label=label,marker=mark_type,color=c)
        Leg_han_m.append(line)
        ax5.scatter(N_lit[:len(Rm_lit)],Rm_lit,s=s,marker=mark_type,color=c)
        ax3.scatter(N_lit[:len(Pm_lit)],Pm_lit,s=s,marker=mark_type,color=c)
    else: 
        line = ax1.scatter(N_lit,Re_lit,s=s,label='GOB(23): simulation',marker=mark_type,color=c)
        ax_1_leg.append(line)
        ax5.scatter(N_lit[:len(Rm_lit)],Rm_lit,s=s,label='GOB(23): simulation',marker=mark_type,color=c)
        ax3.scatter(N_lit[:len(Pm_lit)],Pm_lit,s=s,label='GOB(23): simulation',marker=mark_type,color=c)
legend1 = ax1.legend(handles = Leg_han_m,fontsize=8.5, bbox_to_anchor =(0.008, 0.98), ncol = 2)
ax1.add_artist(legend1)

def lit_M():
    Lit = []; Label = []

    Chris11 = [[128,1500,3000,2]]
    Lit.append(Chris11); Label.append('FCS+(11)')

    Chris14 = [[256,4.6,46,10],[256,26,260,10],[512,1600,3200,2],[512,38,380,10],[512,790,7900,10],[1024,1600,8000,5],[1024,1600,16000,10]]
    Lit.append(Chris14); Label.append('FSB+(14)')

    Amit21 = [[512,2000,2000,1],[512,2000,6000,3]]
    Lit.append(Amit21); Label.append('SF(21)')

    return Lit, Label

Lit, Lit_label = lit_M(); Leg_han_M = []
marker_type = ['*',7,'x']
color = ['green','magenta','turquoise']
for i in range(len(Lit)):
    Data = Lit[i]; N_lit = []; Re_lit = []; Rm_lit = []; Pm_lit = []
    label = Lit_label[i]; mark_type = marker_type[i]; c=color[i]
    for i in range(len(Data)):
        Dat = Data[i]; N_lit.append(Dat[0]); Re_lit.append(Dat[1]); Rm_lit.append(Dat[2]); Pm_lit.append(Dat[3])
    line = ax2.scatter(N_lit,Re_lit,s=s,label=label,marker=mark_type,color=c)
    Leg_han_M.append(line)
    ax6.scatter(N_lit,Rm_lit,s=s,marker=mark_type,color=c)
    ax4.scatter(N_lit,Pm_lit,s=s,marker=mark_type,color=c)
legend1 = ax2.legend(handles = Leg_han_M,fontsize=fontsize, loc='upper left')
ax2.add_artist(legend1)

#============================================== RE ===============================================
Ax = [ax1,ax2]; model = log_Re_of_N; ax_2_leg = []
for i in range(len(Mach)):
    Re_med = Re_Med[i]; Re_err_down = Re_Err_down[i]; Re_err_up =  Re_Err_up[i]
    ax = Ax[i]; p_val = get_p_val(Mach[i]); m = Mach[i]
    Re_asymm_err = np.array(list(zip(Re_err_down, Re_err_up))).T

    line = add_errorbar(ax = ax, y = Re_med, y_err = Re_asymm_err, N_res = N_res)
    if ax == ax1:
        temp = ax_1_leg[0]
        ax_1_leg[0] = line
    else: ax_2_leg.append(line)

    #Fitting: (OURS)
    if m<1: params_ = Params_nfre[m]
    if m>1: params_ = Params_bfre[m]

    y, label = plot_creat_ret(model, params_, N_res)
    line = ax.plot(N_res,y,'-',color='black',label=label)
    if ax == ax1: 
        ax_1_leg.append(line[0])
        ax_1_leg.append(temp)
    else: ax_2_leg.append(line[0])
    
    #Fitting: (GRETA)
    if m<1: 
        my_params = {
            'N_Re': [0.5, 1.0, 5],
            'p_Re': [None, 1.22, None]
        }

        params_alt = perform_fit(model, N_alt, Re_alt, None, my_params)
        y, label = plot_creat_ret(model, params_alt, N_alt)
        line = ax.plot(N_alt,y,'--',color='indigo',label=label)
        ax_1_leg.append(line[0])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([5,5e5])
    if ax == ax1: 
        ax.set_ylabel(r'Re')
        legend2 = ax1.legend(handles = ax_1_leg,fontsize=fontsize, loc=location, markerfirst=markerfirst)
        ax1.add_artist(legend2)
    else:
        legend2 = ax2.legend(handles = ax_2_leg,fontsize=fontsize, loc=location, markerfirst=markerfirst)
        ax2.add_artist(legend2)

# ============================================== PM ===============================================
Ax = [ax3,ax4]; model = log_Pm_form
for i in range(len(Mach)):
    Pm_med = Pm_Med[i]; Pm_err_down = Pm_Err_down[i]; Pm_err_up = Pm_Err_up[i]
    ax = Ax[i]; p_val = get_p_val(Mach[i]); m = Mach[i]
    Pm_asymm_err = np.array(list(zip(Pm_err_down, Pm_err_up))).T

    #Data & Errorbar:
    add_errorbar(ax = ax, y = Pm_med, y_err = Pm_asymm_err, N_res = N_res)

    if m<1: params_ = Params_nfre[m]
    if m>1: params_ = Params_bfre[m]

    N_Re, N_Re_err = get_avg_err_ret(params_['N_Re'])
    p_Re, p_Re_err = get_avg_err_ret(params_['p_Re'])
    N_Rm, N_Rm_err = get_avg_err_ret(params_['N_Rm'])
    p_Rm, p_Rm_err = get_avg_err_ret(params_['p_Rm'])

    params_ = {
        'Pm_0': Pm0_err(N_Re, N_Rm, p_Re, p_Rm, N_Re_err, N_Rm_err, p_Re_err, p_Rm_err),
        'p_Pm': pPm_err(p_Re, p_Rm, p_Re_err, p_Rm_err)
    }
    y, label = plot_creat_ret(model, params_, N_res)
    ax.plot(N_res,y,'-',color='black',label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    if ax == Ax[0]: ax.set_ylabel(r'Pm')
    else:
        order = [1,0]
        handles, labels = fig.get_axes()[3].get_legend_handles_labels()
        ax4.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=fontsize, loc=location, markerfirst=markerfirst)

# ============================================== RM ===============================================
Ax = [ax5,ax6]
for i in range(len(Mach)):
    Rm_med = Rm_Med[i]; Rm_err_down = Rm_Err_down[i]; Rm_err_up = Rm_Err_up[i]
    ax = Ax[i]; p_val = get_p_val(Mach[i]); m = Mach[i]
    Rm_asymm_err = np.array(list(zip(Rm_err_down, Rm_err_up))).T
    order = [1,0]; model = log_Rm_of_N

    #Data & Errorbar:
    add_errorbar(ax = ax, y = Rm_med, y_err = Rm_asymm_err, N_res = N_res)

    if m<1: params_ = Params_nfre[m]
    if m>1: params_ = Params_bfre[m]
    y, label = plot_creat_ret(model, params_, N_res)  
    ax.plot(N_res,y,'-',color='black',label=label)

    #Fitting: (GRETA)
    if m<1: 
        order = [3,1,0,2]

        my_params = {
            'N_Rm': [0.8, 1.5, 5],
            'p_Rm': [None, 1.34, None]
        }
        
        params_ = perform_fit(model, N_alt, Rm_alt, None, my_params)
        params_alt['N_Rm'] = params_['N_Rm']
        params_alt['p_Rm'] = params_['p_Rm']
        y, label = plot_creat_ret(model, params_alt, N_alt)
        ax.plot(N_alt,y,'--',color='indigo',label=label)

        model = log_Pm_form
        N_Re, N_Re_err = get_avg_err_ret(params_alt['N_Re'])
        p_Re, p_Re_err = get_avg_err_ret(params_alt['p_Re'])
        N_Rm, N_Rm_err = get_avg_err_ret(params_alt['N_Rm'])
        p_Rm, p_Rm_err = get_avg_err_ret(params_alt['p_Rm'])

        params_ = {
            'Pm_0': Pm0_err(N_Re, N_Rm, p_Re, p_Rm, N_Re_err, N_Rm_err, p_Re_err, p_Rm_err),
            'p_Pm': pPm_err(p_Re, p_Rm, p_Re_err, p_Rm_err)
        }
        y, label = plot_creat_ret(model, params_, N_alt)
        ax3.plot(N_alt,y,'--',color='indigo',label=label)

        handles, labels = fig.get_axes()[2].get_legend_handles_labels()
        ax3.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=fontsize, loc=location, markerfirst=markerfirst)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$N$')
    if ax == Ax[0]: ax.set_ylabel(r'Rm')
    ax.set_ylim([1e1,6e5])
    ax.set_xlim([np.min(N_alt)-20,np.max(N_res)+500])
    handles, labels = fig.get_axes()[i+4].get_legend_handles_labels()
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=fontsize, loc=location, markerfirst=markerfirst)

#Mach label
M = r'$\mathcal{M}$'
ax1.text(0.90, 0.92, M+' = 0.1', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,fontsize=15,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))
ax2.text(0.86, 0.92, M+' = 10', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes,fontsize=15,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))

fig.savefig(target+filename,format = "pdf",bbox_inches='tight')
print(f'Saved {filename}')
plt.close()
