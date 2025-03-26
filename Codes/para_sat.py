import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cfpack as cfp
import numpy as np
import lmfit
from cfpack.defaults import *
np.seterr(divide='ignore', invalid='ignore')
import os
from vsN_funcs import *
home_directory = os.path.expanduser( '~' )+'/'

def file_access(N_res, spectype, Mach, spec, sat = True):
    if Mach<1: m = '0p'+ str(int(10*Mach))
    else: m = str(int(Mach))
    folder = 'M'+m+'N'+str(N_res).zfill(4)
    sub_folder = spectype+'_'+m+'_result'
    if sat: PATH = '/scratch/pawsey0810/cfederrath/turb_diss/'+folder+'/resim/spectra/' 
    else: PATH = '/scratch/pawsey0810/cfederrath/turb_diss/'+folder+'/spectra/' 
    file = open(PATH+spec).readlines()
    get_col = lambda col: (line.split()[col-1] for line in file[6:])
    k = np.array(list(map(float,get_col(1))))
    log_E_tot = np.array(list(map(float,get_col(6))))
    log_E_tot_err = np.array(list(map(float,get_col(7))))
    E_tot = 10**log_E_tot 
    E_tot_err_up = 10**(log_E_tot+log_E_tot_err)
    E_tot_err_down = 10**(log_E_tot-log_E_tot_err)
    E_tot_err = (E_tot_err_down+E_tot_err_up)/2

    return k, E_tot, E_tot_err_up, E_tot_err_down, E_tot_err

#color = ['blue','orange','magenta','green','red','purple']
color = ['grey']
M = [0.1,10]
#N_res = [2304,1152,576,288,144]
N_res = [576]
factor = []
for i in range(len(N_res)): factor.append(1/10**i)
factor = 0.5*np.array(factor).astype(float)
space = ['','','','']
folder = home_directory+'Simu_results/'
try: os.makedir(folder)
except: pass

def kin_est(N_res,Mach,A,k_ex,k_cons,k_ex_min,k_ex_max,k_cons_max,sat=True):
    #Defining functions
    spectype = 'kin'
    spec = 'aver_spect_vels.dat'

    A_k = A
    k_nu_val = k_cons
    k_bn_val = k_ex
    k_cons = r'$k_{\nu}$'

    if Mach<1:
        p_k = -1.7
        p_nu_val = 1
        A_k_max = 1e-2
        A_k_min = 1e-6
        p_bn_val = 0.5
    else:
        p_k = -2
        p_nu_val = 0.7
        A_k_max = 1e2
        A_k_min = 1e-5
        p_bn_val = 0.5

    params = lmfit.Parameters()
    params.add('A',value = A, min = A_k_min, max = A_k_max)
    params.add('p_kin',value = p_k)
    params['p_kin'].vary = False
    params.add('p_bn',value = p_bn_val,min=0.0,max = 1.0)
    params.add('k_bn',value = k_bn_val,min=k_ex_min,max=k_ex_max)
    params.add('k_nu',value = k_nu_val, max=k_cons_max)
    params.add('p_nu', value = p_nu_val)
    params['p_nu'].vary = False

    #Accessing files and data
    k, E_tot, E_tot_err_up, E_tot_err_down, E_tot_err = file_access(N_res, spectype, Mach, spec,sat)

    #Fitting
    index = Index(k,N_res)
    weights = 1/E_tot_err[index]

    mod = lmfit.Model(P_kin_fit_eq)
    y_fit = E_tot[index]
    fit = mod.fit(y_fit,params=params,k=k[index],weights=weights)
    #print(fit.fit_report())
    
    y = P_kin_fit_eq(k[index],fit.values['A'],fit.values['p_kin'],fit.values['p_bn'],fit.values['k_bn'],fit.values['k_nu'],fit.values['p_nu'])
    p_nu = fit.values['p_nu']; p_kin = fit.values['p_kin']
    for name, param in fit.params.items():
        if name == 'k_nu':
            k_nu_value = param.value
            k_nu_err = param.stderr
            k_nu_actual = k_nu_value**(1/p_nu)
            k_nu_err_act = k_nu_err*k_nu_actual/(k_nu_value*p_nu)
        if name == 'p_bn':
            p_bn = param.value
            p_bn_err = param.stderr
        if name == 'k_bn':
            k_bn_value = param.value
            k_bn_err = param.stderr
        else: pass
    #print(f'k_nu_actual = {k_nu_actual}')

    return k, E_tot, E_tot_err_up, E_tot_err_down, y, k_nu_value, k_nu_err, k_nu_actual, k_nu_err_act, k_bn_value, k_bn_err, p_kin, p_bn, p_bn_err, p_nu

def mag_est(N_res,Mach,A,k_cons,k_cons_max,p_weight,sat=True):
    spectype = 'mag'
    spec = 'aver_spect_mags.dat'

    A_mag = A
    k_eta_value = k_cons
    A_mag_max = 1e-1
    A_mag_min = 1e-7
    p_mag_val = 1.5
    p_eta_val = 1.0
    k_cons = r'$k_{\eta}$'
    
    params = lmfit.Parameters()
    params.add('A',value = A_mag, min = A_mag_min, max = A_mag_max)
    params.add('p_mag',value = p_mag_val, min=0.5,max=3.0)
    params.add('k_eta',value = k_eta_value, min = 0.0,max= k_cons_max)
    params.add('p_eta', value = p_eta_val, min=0.5, max = 1.3)

    #Accessing files and data
    k, E_tot, E_tot_err_up, E_tot_err_down, E_tot_err = file_access(N_res, spectype, Mach, spec, sat)
 
    index = Index(k,N_res)
    weights = 1/E_tot_err[index]
   
    mod = lmfit.Model(P_mag_fit_eq)
    y_fit = E_tot[index]
    fit = mod.fit(y_fit,params=params,k=k[index],weights=weights)
    print(fit.fit_report())
    
    for name, param in fit.params.items():
        if name == 'k_eta':
            k_eta_value = param.value
            k_eta_err = param.stderr
        if name == 'p_eta':
            p_eta = param.value
            p_eta_err = param.stderr
        if name == 'p_mag':
            p_mag = param.value
            p_mag_err = param.stderr
        else: pass
    
    y = P_mag_fit_eq(k[index],fit.values['A'],fit.values['p_mag'],fit.values['k_eta'],fit.values['p_eta'])
    k_eta_actual = k_eta_value**(1/p_eta)
    c_1 = k_eta_actual*np.log(k_eta_actual)
    c_2 = k_eta_value*np.log(k_eta_value)
    k_eta_err_act = c_1*((p_eta_err/p_eta)+(k_eta_err/c_2))
    print(f'k_eta_actual = {k_eta_actual}')

    return k, E_tot, E_tot_err_up, E_tot_err_down, y, k_eta_value, k_eta_err, k_eta_actual, k_eta_err_act, p_mag, p_mag_err, p_eta, p_eta_err

#A,k_ex,k_cons,k_ex_min,k_ex_max,k_cons_max
def kin_init_para(m):
    if m<1: return np.array([[1e-4,13,19,None,None,None]])
        #return np.array([[1e-5,45,75,None,None,None],[3e-5,25,36,20,None,38],[1e-4,13,19,None,None,None],[1e-4,6,10,None,None,None],[1e-3,3,5,None,None,None],[1e-3,0.2,2.9,0.0,1.0,None],[1e-3,0.1,1.7,0.0,1.0,None]])
    else: return np.array([[4e-1,17,20,None,None,None]])
        #return np.array([[1e-1,30,40,None,None,None],[1e-1,20,40,15,30,None],[4e-1,17,20,None,None,None],[1.5,8,10,5,None,None],[5,4,5,None,None,None],[14,2,2.5,0.5,None,None],[20,1,2,0.5,None,None]])

def mag_init_para(m):
    if m<1: return np.array([[2.3e-5,18,None,-0.5]])
        #return np.array([[1e-6,68,None,-0.5],[3e-6,34,None,-0.5],[2.3e-5,18,None,-0.5],[1.8e-4,10.1,None,-0.5],[1.3e-3,2.4,None,-1],[1e-2,2.5,None,-1]])
    else: return np.array([[3e-4,32,None,-0.5]])
        #return np.array([[2e-5,70,None,-0.4],[8e-5,55,None,-0.5],[3e-4,32,None,-0.5],[9e-4,20.1,None,-0.5],[3e-3,12,None,-0.5],[1e-2,7,None,-2.5]])

k_kin = []; E_tot_kin = []; E_tot_err_up_kin = []; E_tot_err_down_kin = []; y_kin = []
k_mag = []; E_tot_mag = []; E_tot_err_up_mag = []; E_tot_err_down_mag = []; y_mag = [] 
K_nu = []; K_eta = []

for Mach in M:
    k_nu = space+['k_nu']; k_nu_er = space+['k_nu_error']; k_nu_act = space+['k_nu_act']; k_nu_er_act = space+['k_nu_act_error']
    k_bn = space+['k_bn']; k_bn_er = space+['k_bn_error']
    P_kin = space+['p_kin']; P_bn = space+['p_bn']; P_bn_err = space+['p_bn_error']; P_nu = space+['p_nu']
    k_eta = space+['k_eta']; k_eta_er = space+['k_eta_error']; k_eta_act = space+['k_eta_act']; k_eta_er_act = space+['k_eta_act_error']
    P_mag = space+['p_mag']; P_mag_err = space+['p_mag_error']; P_eta = space+['p_eta']; P_eta_err = space+['p_eta_error']

    if Mach<1: m = '0p'+ str(int(10*Mach))
    else: m = str(int(Mach))

    for i,N in enumerate(N_res):
        init = kin_init_para(Mach)[i]
        A = init[0]; k_ex = init[1]; k_cons = init[2]; k_ex_min=init[3]; k_ex_max=init[4];k_cons_max = init[5]
        k, E_tot, E_tot_err_up, E_tot_err_down, Y,k_nu_value,k_nu_err,k_nu_actual,k_nu_err_act,k_bn_value,k_bn_err,p_kin,p_bn,p_bn_err,p_nu = kin_est(N,Mach,A,k_ex,k_cons,k_ex_min,k_ex_max,k_cons_max)

        k_kin.append(k); E_tot_kin.append(E_tot); E_tot_err_up_kin.append(E_tot_err_up); E_tot_err_down_kin.append(E_tot_err_down)
        y_kin.append(Y); k_nu.append(k_nu_value); k_nu_er.append(k_nu_err); k_nu_act.append(k_nu_actual); k_nu_er_act.append(k_nu_err_act)
        k_bn.append(k_bn_value); k_bn_er.append(k_bn_err); P_kin.append(p_kin); P_bn_err.append(p_bn_err); P_bn.append(p_bn); P_nu.append(p_nu)
        K_nu.append(k_nu_actual)

        init = mag_init_para(Mach)[i]
        A = init[0]; k_cons = init[1]; k_cons_max = init[2]; p_weight = init[3]
        K,E_Tot,E_Tot_err_up,E_Tot_err_down,Y,k_eta_value,k_eta_err,k_eta_actual,k_eta_err_act,p_mag,p_mag_err,p_eta,p_eta_err = mag_est(N,Mach,A,k_cons,k_cons_max,p_weight)

        k_mag.append(K); E_tot_mag.append(E_Tot); E_tot_err_up_mag.append(E_Tot_err_up); E_tot_err_down_mag.append(E_Tot_err_down); y_mag.append(Y)
        k_eta.append(k_eta_value); k_eta_er.append(k_eta_err); k_eta_act.append(k_eta_actual); k_eta_er_act.append(k_eta_err_act)
        P_mag.append(p_mag); P_mag_err.append(p_mag_err); P_eta.append(p_eta); P_eta_err.append(p_eta_err)
        K_eta.append(k_eta_actual)

    N_re = space+['N_res']+N_res; N_re = np.array(N_re)
    k_nu_act = np.array(k_nu_act); k_nu_er_act = np.array(k_nu_er_act)
    k_nu = np.array(k_nu); k_nu_er = np.array(k_nu_er)
    k_bn = np.array(k_bn); k_bn_er = np.array(k_bn_er)
    P_kin = np.array(P_kin); P_bn_err = np.array(P_bn_err)
    P_bn = np.array(P_bn); P_nu = np.array(P_nu)

    filename = 'k_nu_vs_N_'+m+'sat.txt'               
    datafile_path = folder+filename
    data_kin = np.column_stack([N_re, P_kin, P_bn, P_bn_err, P_nu, k_bn, k_bn_er, k_nu, k_nu_er, k_nu_act, k_nu_er_act])
    np.savetxt(datafile_path , data_kin, delimiter = '\t',fmt = '%s')
    print(f'Saved{filename}')

    k_eta_act = np.array(k_eta_act); k_eta_er_act = np.array(k_eta_er_act)
    k_eta = np.array(k_eta); k_eta_er = np.array(k_eta_er)
    P_mag = np.array(P_mag); P_mag_err = np.array(P_mag_err)
    P_eta = np.array(P_eta); P_eta_err = np.array(P_eta_err)

    filename = 'k_eta_vs_N_'+m+'sat.txt'
    datafile_path = folder+filename
    data_mag = np.column_stack([N_re, P_mag, P_mag_err, P_eta, P_eta_err, k_eta, k_eta_er, k_eta_act, k_eta_er_act])
    np.savetxt(datafile_path , data_mag, delimiter = '\t',fmt = '%s')
    print(f'Saved{filename}')

#Plotting
fig = plt.figure(figsize = (20,12))
plt.rcParams.update({'font.size': 20})
location = 'upper left'; coord = (0.69, 0.91); labelspacing = 0.1
linewidth = 1.0; linewidth_data = 2.0; alpha = 0.3; fontsize = 21
Mach_label = r'$\mathcal{M}$'
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
plt.subplots_adjust(wspace=0.01, hspace=0.01)

N_len = len(N_res); K_nu = np.array(K_nu); K_eta = np.array(K_eta)
for i in np.arange(N_len):
    n = N_res[i]; c = color[i]

    #Axis 1
    k = k_kin[i]; y = factor[i]*E_tot_kin[i]; y_down = factor[i]*E_tot_err_down_kin[i]; y_up = factor[i]*E_tot_err_up_kin[i]
    y_fit = factor[i]*y_kin[i]; index = Index(k,n); x = k; ylab = Pkin_label; k_label = knu_label

    ax1.plot(x, y,'-', color=c, linewidth=linewidth_data, label='Saturation')
    ax1.fill_between(x, y_down, y_up, color=c, alpha=alpha)
    ax1.plot(x[index], y_fit,'--',linewidth=linewidth,color='black')
    ax1.set_ylim([1e-10,5e1])
    ax1.legend(bbox_to_anchor = coord, labelspacing=labelspacing, fontsize=fontsize)
    ax1.axvline(x = K_nu[i], ymax=0.035, color=c, linewidth = 3.0)
    ax1.set_ylabel(ylab); ax1.set_xscale('log'); ax1.set_yscale('log')
    axis_to_data = ax1.transAxes + ax1.transData.inverted()
    k_coord = axis_to_data.inverted().transform((K_nu[i],6e-10))
    ax1.text(k_coord[0]-0.01,k_coord[1], k_label, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,fontsize=fontsize,color=c)
    
    #Axis 2
    k = k_kin[i+N_len]; y = factor[i]*E_tot_kin[i+N_len]; y_down = factor[i]*E_tot_err_down_kin[i+N_len]; y_up = factor[i]*E_tot_err_up_kin[i+N_len]
    y_fit = factor[i]*y_kin[i+N_len]; index = Index(k,n); x = k

    ax2.plot(x, y,'-', color=c, linewidth=linewidth_data, label='Saturation')
    ax2.fill_between(x, y_down, y_up, color=c, alpha=alpha)
    ax2.plot(x[index], y_fit,'--',linewidth=linewidth,color='black')
    ax2.legend(bbox_to_anchor = coord, labelspacing=labelspacing, fontsize=fontsize)
    ax2.axvline(x = K_nu[i+N_len], ymax=0.035, color=c, linewidth = 3.0)
    ax2.set_xscale('log'); ax2.set_yscale('log')
    axis_to_data = ax2.transAxes + ax2.transData.inverted()
    k_coord = axis_to_data.inverted().transform((K_nu[i+N_len],6e-10))
    ax2.text(k_coord[0]-0.02,k_coord[1], k_label, horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes,fontsize=fontsize,color=c)
    
    #Axis 3
    k = k_mag[i]; y = factor[i]*E_tot_mag[i]; y_down = factor[i]*E_tot_err_down_mag[i]; y_up = factor[i]*E_tot_err_up_mag[i]
    y_fit = factor[i]*y_mag[i]; index = Index(k,n); x = k; ylab = Pmag_label; k_label = keta_label

    ax3.plot(x, y,'-', color=c, linewidth=linewidth_data, label='Saturation')
    ax3.fill_between(x, y_down, y_up, color=c, alpha=alpha)
    ax3.plot(x[index], y_fit,'--',linewidth=linewidth,color='black')
    ax3.axvline(x = K_eta[i], ymax=0.035, color=c, linewidth = 3.0)
    ax3.set_ylim([1e-7,4e-2])
    ax3.set_xlabel(xlab); ax3.set_ylabel(ylab)
    ax3.set_xscale('log'); ax3.set_yscale('log')
    axis_to_data = ax3.transAxes + ax3.transData.inverted()
    k_coord = axis_to_data.inverted().transform((K_eta[i],3e-7))
    ax3.text(k_coord[0],k_coord[1], k_label, horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes,fontsize=fontsize,color=c)
    
    #Axis 4
    k = k_mag[i+N_len]; y = factor[i]*E_tot_mag[i+N_len]; y_down = factor[i]*E_tot_err_down_mag[i+N_len]; y_up = factor[i]*E_tot_err_up_mag[i+N_len]
    y_fit = factor[i]*y_mag[i+N_len]; index = Index(k,n); x = k

    ax4.plot(x, y,'-', color=c, linewidth=linewidth_data, label='Saturation')
    ax4.fill_between(x, y_down, y_up, color=c, alpha=alpha)
    ax4.plot(x[index], y_fit,'--',linewidth=linewidth,color='black')
    ax4.axvline(x = K_eta[i+N_len], ymax=0.035, color=c, linewidth = 3.0)
    ax4.set_xlabel(xlab); ax4.set_xscale('log'); ax4.set_yscale('log')
    axis_to_data = ax3.transAxes + ax3.transData.inverted()
    k_coord = axis_to_data.inverted().transform((K_eta[i+N_len],3e-7))
    ax4.text(k_coord[0]+0.02,k_coord[1], k_label, horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes,fontsize=fontsize,color=c)
 
#Mach label
ax1.text(0.87, 0.93, Mach_label+' = 0.1', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,fontsize=21,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))
ax2.text(0.87, 0.93, Mach_label+' = 10', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes,fontsize=21,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))

#*************************************** Adding kinematic regime *******************************************

k_kin = []; E_tot_kin = []; E_tot_err_up_kin = []; E_tot_err_down_kin = []; y_kin = []
k_mag = []; E_tot_mag = []; E_tot_err_up_mag = []; E_tot_err_down_mag = []; y_mag = [] 
K_nu = []; K_eta = []

for Mach in M:
    if Mach<1: m = '0p'+ str(int(10*Mach))
    else: m = str(int(Mach))

    for i,N in enumerate(N_res):
        init = kin_init_para(Mach)[i]
        A = init[0]; k_ex = init[1]; k_cons = init[2]; k_ex_min=init[3]; k_ex_max=init[4];k_cons_max = init[5]
        k, E_tot, E_tot_err_up, E_tot_err_down, Y,k_nu_value,k_nu_err,k_nu_actual,k_nu_err_act,k_bn_value,k_bn_err,p_kin,p_bn,p_bn_err,p_nu = kin_est(N,Mach,A,k_ex,k_cons,k_ex_min,k_ex_max,k_cons_max,sat=False)

        k_kin.append(k); E_tot_kin.append(E_tot); E_tot_err_up_kin.append(E_tot_err_up); E_tot_err_down_kin.append(E_tot_err_down)
        y_kin.append(Y); K_nu.append(k_nu_actual)

        init = mag_init_para(Mach)[i]
        A = init[0]; k_cons = init[1]; k_cons_max = init[2]; p_weight = init[3]
        K,E_Tot,E_Tot_err_up,E_Tot_err_down,Y,k_eta_value,k_eta_err,k_eta_actual,k_eta_err_act,p_mag,p_mag_err,p_eta,p_eta_err = mag_est(N,Mach,A,k_cons,k_cons_max,p_weight,sat=False)

        k_mag.append(K); E_tot_mag.append(E_Tot); E_tot_err_up_mag.append(E_Tot_err_up); E_tot_err_down_mag.append(E_Tot_err_down)
        y_mag.append(Y); K_eta.append(k_eta_actual)

K_nu = np.array(K_nu); K_eta = np.array(K_eta); color = ['magenta']; coord = (0.69, 0.88)
for i in np.arange(N_len):
    n = N_res[i]; c = color[i]

    #Axis 1
    k = k_kin[i]; y = factor[i]*E_tot_kin[i]; y_down = factor[i]*E_tot_err_down_kin[i]; y_up = factor[i]*E_tot_err_up_kin[i]
    y_fit = factor[i]*y_kin[i]; index = Index(k,n); x = k; k_label = knu_label

    ax1.plot(x, y,'-', color=c, linewidth=linewidth_data, label='Kinematic')
    ax1.fill_between(x, y_down, y_up, color=c, alpha=alpha)
    ax1.legend(bbox_to_anchor = coord, labelspacing=labelspacing, fontsize=fontsize)
    ax1.axvline(x = K_nu[i], ymax=0.035, color=c, linewidth = 3.0)
    axis_to_data = ax1.transAxes + ax1.transData.inverted()
    k_coord = axis_to_data.inverted().transform((K_nu[i],6e-10))
    ax1.text(k_coord[0]+0.01,k_coord[1], k_label, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,fontsize=fontsize,color=c)
    
    #Axis 2
    k = k_kin[i+N_len]; y = factor[i]*E_tot_kin[i+N_len]; y_down = factor[i]*E_tot_err_down_kin[i+N_len]; y_up = factor[i]*E_tot_err_up_kin[i+N_len]
    y_fit = factor[i]*y_kin[i+N_len]; index = Index(k,n); x = k

    ax2.plot(x, y,'-', color=c, linewidth=linewidth_data, label='Kinematic')
    ax2.fill_between(x, y_down, y_up, color=c, alpha=alpha)
    ax2.legend(bbox_to_anchor = coord, labelspacing=labelspacing, fontsize=fontsize)
    ax2.axvline(x = K_nu[i+N_len], ymax=0.035, color=c, linewidth = 3.0)
    axis_to_data = ax2.transAxes + ax2.transData.inverted()
    k_coord = axis_to_data.inverted().transform((K_nu[i+N_len],6e-10))
    ax2.text(k_coord[0]+0.02,k_coord[1], k_label, horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes,fontsize=fontsize,color=c)
    
    #Axis 3
    k = k_mag[i]; y = factor[i]*E_tot_mag[i]; y_down = factor[i]*E_tot_err_down_mag[i]; y_up = factor[i]*E_tot_err_up_mag[i]
    y_fit = factor[i]*y_mag[i]; index = Index(k,n); x = k; k_label = keta_label

    ax3.plot(x, y,'-', color=c, linewidth=linewidth_data, label='Kinematic')
    ax3.fill_between(x, y_down, y_up, color=c, alpha=alpha)
    ax3.axvline(x = K_eta[i], ymax=0.035, color=c, linewidth = 3.0)
    axis_to_data = ax3.transAxes + ax3.transData.inverted()
    k_coord = axis_to_data.inverted().transform((K_eta[i],3e-7))
    ax3.text(k_coord[0],k_coord[1], k_label, horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes,fontsize=fontsize,color=c)
    
    #Axis 4
    k = k_mag[i+N_len]; y = factor[i]*E_tot_mag[i+N_len]; y_down = factor[i]*E_tot_err_down_mag[i+N_len]; y_up = factor[i]*E_tot_err_up_mag[i+N_len]
    y_fit = factor[i]*y_mag[i+N_len]; index = Index(k,n); x = k

    ax4.plot(x, y,'-', color=c, linewidth=linewidth_data, label='Kinematic')
    ax4.fill_between(x, y_down, y_up, color=c, alpha=alpha)
    ax4.axvline(x = K_eta[i+N_len], ymax=0.035, color=c, linewidth = 3.0)
    axis_to_data = ax3.transAxes + ax3.transData.inverted()
    k_coord = axis_to_data.inverted().transform((K_eta[i+N_len],3e-7))
    ax4.text(k_coord[0]-0.02,k_coord[1], k_label, horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes,fontsize=fontsize,color=c)

filename = 'spectra_sat.pdf'
plt.savefig(folder+filename,format = "pdf")
plt.close()
print(f'Saved {filename}')