import matplotlib, lmfit, os, random
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cfpack as cfp
from cfpack.defaults import *
import numpy as np
#from Utilites import *
from vsN_funcs import *
np.seterr(divide='ignore', invalid='ignore')
home_directory = os.path.expanduser( '~' )+'/'

def file_access(N_res, spectype, Mach, spec):
    if isinstance(Mach,float): 
        m = str(Mach).split(".")
        m = m[0] + 'p' + m[1]
    else: m = str(int(Mach))
    folder = 'M'+m+'N'+str(N_res).zfill(4)
    PATH = '/scratch/pawsey0810/cfederrath/turb_diss/'+folder+'/spectra/'
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

color = ['blue','orange','magenta','green','red','purple','gray','brown','turquoise','indigo','deeppink','maroon','olivedrab','gold','cyan','sandybrown']
M = [160, 80, 40, 20, 10, 5, 2.5, 1.25, 0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]
#M = [0.8, 0.4, 0.2, 0.1, 0.05]
#M = [20, 10, 5, 2.5, 1.25]
#M = [0.00625]

factor = [10**(-i) for i in range(len(M))]
factor = 0.5*np.array(factor).astype(float)
N_res = 576
space = ['','','','']

def kin_est(N_res,Mach,A,k_ex,k_cons,k_ex_min,k_ex_max,k_cons_max):
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
        A_k_min = 1e-8
    elif Mach == 1.25:
        p_k = -2
        p_nu_val = 1
        A_k_max = 1
        A_k_min = 1e-4
    else:
        p_k = -2
        p_nu_val = 0.7
        A_k_max = 1e3
        A_k_min = 1e-6
    p_bn_val = 0.5

    params = lmfit.Parameters()
    params.add('p_kin',value = p_k)
    params['p_kin'].vary = False
    '''if Mach == 1.25:
        params.add('A',value = A, min = 1e-3, max = 1e-1) 
        params.add('p_bn',value = 1.5e-11,min=8e-12,max = 2e-11)
        params.add('k_nu',value = k_nu_val, min=k_cons_max)'''
    params.add('A',value = A, min = A_k_min, max = A_k_max)
    params.add('p_bn',value = p_bn_val,min=0.0,max = 1.0)
    params.add('k_nu',value = k_nu_val, max=k_cons_max)
    params.add('k_bn',value = k_bn_val,min=k_ex_min,max=k_ex_max)
    params.add('p_nu', value = p_nu_val)
    params['p_nu'].vary = False

    #Accessing files and data
    k, E_tot, E_tot_err_up, E_tot_err_down, E_tot_err = file_access(N_res, spectype, Mach, spec)

    #Fitting
    index = Index(k,N_res)
    weights = 1/E_tot_err[index]

    mod = lmfit.Model(P_kin_fit_eq)
    y_fit = E_tot[index]
    fit = mod.fit(y_fit,params=params,k=k[index],weights=weights)
    print(fit.fit_report())

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
    print(Mach)
    return k, E_tot, E_tot_err_up, E_tot_err_down, y, k_nu_value, k_nu_err, k_nu_actual, k_nu_err_act, k_bn_value, k_bn_err, p_kin, p_bn, p_bn_err, p_nu

def mag_est(N_res,Mach,A,k_cons,k_cons_max,p_weight):
    print('...')
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
    params.add('p_mag',value = p_mag_val, min=1.0,max=3.5)
    params.add('k_eta',value = k_eta_value, min = 0.0,max= k_cons_max)
    params.add('p_eta', value = p_eta_val, min=0.5, max = 1.3)

    #Accessing files and data
    k, E_tot, E_tot_err_up, E_tot_err_down, E_tot_err = file_access(N_res, spectype, Mach, spec)
 
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
    #print(f'k_eta_actual = {k_eta_actual}')

    return k, E_tot, E_tot_err_up, E_tot_err_down, y, k_eta_value, k_eta_err, k_eta_actual, k_eta_err_act, p_mag, p_mag_err, p_eta, p_eta_err

#A,k_ex,k_cons,k_ex_min,k_ex_max,k_cons_max
kin_init_para = {
0.00625: [4e-8,10,14,5,12,20],
0.0125: [1.6e-7,10,15,5,12,20],
0.025: [6.25e-6,10,16,None,None,None],
0.05: [2.5e-5,10,17,None,None,None],
0.1: [1e-4,11,18,None,None,None],
0.2: [4e-4,12,19,None,None,None],
0.4: [16e-4,13,20,None,None,None],
0.8: [64e-4,13,21,None,None,None],
#Supersonic
1.25: [4e-2,13,22,10,14,26],
2.5: [8e-2,10.5,10,None,None,None],
5: [2e-1,12.5,10,None,None,None],
10: [1,10.5,10,None,None,None],
20: [5,9,10,None,None,None],
40: [25,8,10,None,None,None],
80: [100,8,10,None,None,None],
160: [400,8,10,None,None,None]
}
#A,k_cons,k_cons_max,p_weight
mag_init_para = {
0.00625: [4.5e-5,1,5,-0.5],
0.0125: [4e-5,1,None,-0.5],
0.025: [3.5e-5,3,None,-0.5],
0.05: [3e-5,8,None,-0.5],
0.1: [2.5e-5,13,None,-0.5],
0.2: [2e-5,18,None,-0.5],
0.4: [1.5e-5,21,None,-0.5],
0.8: [2.5e-5,24,None,-0.5],
#Supersonic
1.25: [3e-5,26,None,-0.5],
2.5: [1.5e-4,22,None,-0.5],
5: [4e-4,18,None,-0.5],
10: [3e-4,23,None,-0.5],
20: [1.2e-4,28,None,-0.5],
40: [6e-5,26,None,-0.5],
80: [5e-5,21,None,-0.5],
160: [4e-5,21,None,-0.5]
}

k_kin = []; E_tot_kin = []; E_tot_err_up_kin = []; E_tot_err_down_kin = []; y_kin = []
k_mag = []; E_tot_mag = []; E_tot_err_up_mag = []; E_tot_err_down_mag = []; y_mag = [] 

folder = 'Simu_results'
k_nu = ['k_nu']; k_nu_er = ['k_nu_error']; k_nu_act = ['k_nu_act']; k_nu_er_act = ['k_nu_act_error']
k_bn = ['k_bn']; k_bn_er = ['k_bn_error']
P_kin = ['p_kin']; P_bn = ['p_bn']; P_bn_err = ['p_bn_error']; P_nu = ['p_nu']
k_eta = ['k_eta']; k_eta_er = ['k_eta_error']; k_eta_act = ['k_eta_act']; k_eta_er_act = ['k_eta_act_error']
P_mag = ['p_mag']; P_mag_err = ['p_mag_error']; P_eta = ['p_eta']; P_eta_err = ['p_eta_error']

for Mach in M:
    init = kin_init_para[Mach]
    A = init[0]; k_ex = init[1]; k_cons = init[2]; k_ex_min=init[3]; k_ex_max=init[4];k_cons_max = init[5]
    K,E_Tot,E_Tot_err_up,E_Tot_err_down,Y,k_nu_value,k_nu_err,k_nu_actual,k_nu_err_act,k_bn_value,k_bn_err,p_kin,p_bn,p_bn_err,p_nu = kin_est(N_res,Mach,A,k_ex,k_cons,k_ex_min,k_ex_max,k_cons_max)

    k_kin.append(K); E_tot_kin.append(E_Tot); E_tot_err_up_kin.append(E_Tot_err_up); E_tot_err_down_kin.append(E_Tot_err_down)
    y_kin.append(Y); k_nu.append(k_nu_value); k_nu_er.append(k_nu_err); k_nu_act.append(k_nu_actual); k_nu_er_act.append(k_nu_err_act)
    k_bn.append(k_bn_value); k_bn_er.append(k_bn_err); P_kin.append(p_kin); P_bn_err.append(p_bn_err); P_bn.append(p_bn); P_nu.append(p_nu)

    init = mag_init_para[Mach]
    A = init[0]; k_cons = init[1]; k_cons_max = init[2]; p_weight = init[3]
    K,E_Tot,E_Tot_err_up,E_Tot_err_down,Y,k_eta_value,k_eta_err,k_eta_actual,k_eta_err_act,p_mag,p_mag_err,p_eta,p_eta_err = mag_est(N_res,Mach,A,k_cons,k_cons_max,p_weight)

    k_mag.append(K); E_tot_mag.append(E_Tot); E_tot_err_up_mag.append(E_Tot_err_up); E_tot_err_down_mag.append(E_Tot_err_down); y_mag.append(Y)
    k_eta.append(k_eta_value); k_eta_er.append(k_eta_err); k_eta_act.append(k_eta_actual); k_eta_er_act.append(k_eta_err_act)
    P_mag.append(p_mag); P_mag_err.append(p_mag_err); P_eta.append(p_eta); P_eta_err.append(p_eta_err)

filename = 'k_nu_vs_mach_'+f'{N_res}'+'.txt'               
datafile_path = home_directory+folder+'/'+filename
data_kin = np.column_stack([space+['Mach']+M, space+P_kin, space+P_bn, space+P_bn_err, space+P_nu, space+k_bn, space+k_bn_er, space+k_nu, space+k_nu_er, space+k_nu_act, space+k_nu_er_act])
np.savetxt(datafile_path , data_kin, delimiter = '\t',fmt = '%s')
print(f'Saved{filename}')

filename = 'k_eta_vs_mach_'+f'{N_res}'+'.txt'
datafile_path = home_directory+folder+'/'+filename
data_mag = np.column_stack([space+['Mach']+M, space+P_mag, space+P_mag_err, space+P_eta, space+P_eta_err, space+k_eta, space+k_eta_er, space+k_eta_act, space+k_eta_er_act])
np.savetxt(datafile_path , data_mag, delimiter = '\t',fmt = '%s')
print(f'Saved{filename}')

#Plotting
plt.rcParams["figure.figsize"] = [20, 12]
plt.rcParams.update({'font.size': 20})
location = 'upper left'; coord = (0.012, 0.95); labelspacing = 0.07
linewidth = 1.0; linewidth_data = 2.0; alpha = 0.3; 
mach = r'$\mathcal{M}$'
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)
#sharing x
ax3.sharex(ax1); ax4.sharex(ax2)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.subplots_adjust(wspace=0.1, hspace=0.03)

K_nu = np.array(k_nu_act[1:]).astype(float); K_eta = np.array(k_eta_act[1:]).astype(float)
K_nu_err = np.array(k_nu_er_act[1:]).astype(float); K_eta_err = np.array(k_eta_er_act[1:]).astype(float)

for i in range(len(M)):
    m = M[i]; c = color[i]

    #Axis 1
    k = k_kin[i]; y = factor[i]*E_tot_kin[i]; y_fit = factor[i]*y_kin[i]
    y_down = factor[i]*E_tot_err_down_kin[i]; y_up = factor[i]*E_tot_err_up_kin[i]
    index = Index(k,N_res); x = k

    ax1.plot(x, y,'-', color=c, linewidth=linewidth_data, label=str(m))
    ax1.fill_between(x, y_down, y_up, color=c, alpha=alpha)
    ax1.plot(x[index], y_fit,'-',linewidth=linewidth,color='black')
    ax1.legend(bbox_to_anchor = coord, labelspacing=labelspacing, fontsize=15)
    ax1.axvline(x = K_nu[i], ymax=0.035, color=c, linewidth = 3.0)
    
    #Axis 3
    k = k_mag[i]; y = factor[i]*E_tot_mag[i]; y_fit = factor[i]*y_mag[i]
    y_down = factor[i]*E_tot_err_down_mag[i]; y_up = factor[i]*E_tot_err_up_mag[i]
    index = Index(k,N_res); x = k

    ax3.plot(x, y,'-', color=c, linewidth=linewidth_data, label=str(m))
    ax3.fill_between(x, y_down, y_up, color=c, alpha=alpha)
    ax3.plot(x[index], y_fit,'-',linewidth=linewidth,color='black')
    ax3.axvline(x = K_eta[i], ymax=0.035, color=c, linewidth = 3.0)
    #ax3.set_ylim([1e-15,1e-2])

ylab = r'$P_{\rm '+'kin'+'}$'; ax1.set_ylabel(ylab); ax1.set_xscale('log'); ax1.set_yscale('log')
ylab = r'$P_{\rm '+'mag'+'}$'; ax3.set_xlabel(xlab); ax3.set_ylabel(ylab)
ax3.set_xscale('log'); ax3.set_yscale('log')

#Nres label
ax1.text(0.07, 0.95, f'{mach}', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,fontsize=14,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))
#K labels
knu_label = r'$k_{\nu}$'
ax1.text(0.65, 0.07, knu_label, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,fontsize=20,color='black')
keta_label = r'$k_{\eta}$'
ax3.text(0.7, 0.07, keta_label, horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes,fontsize=20,color='black')
# ************************************************* k vs M ******************************************************

ax2.axvline(x=1,ymax=30,linestyle='dashed',color='k')
ax4.axvline(x=1,ymax=53,linestyle='dashed',color='k')
ax2.plot(M, K_nu,'-',linewidth=linewidth,color='black')
add_errorbar(ax2, K_nu, K_nu_err, M)
ax4.plot(M, K_eta,'-',linewidth=linewidth,color='black')
add_errorbar(ax4, K_eta, K_eta_err, M)
ax2.set_ylabel(knu_label); ax4.set_ylabel(keta_label); ax4.set_xlabel(r'$\mathcal{M}$')
ax2.set_ylim([14.5,30]); ax4.set_ylim([-1,53]); ax1.set_ylim([1e-28,1e7]); ax3.set_xlim([0.3,4e2])
#ax2.set_yscale('log'); ax4.set_yscale('log')
ax2.set_xscale('log'); ax4.set_xscale('log')
fig = plt.gcf(); fig.suptitle('$N = $'+f' {N_res}', fontsize=20, y = 0.92)
filename = home_directory+folder+'/'+'spectra_mach.pdf'
plt.savefig(filename,format = "pdf")
plt.close()

# ************************************************* R vs M ******************************************************
#Plotting
plt.rcParams["figure.figsize"] = [8, 20]
plt.rcParams.update({'font.size': 20})
location = 'upper left'; coord = (0.75, 0.91); labelspacing = 0.1
linewidth = 1.0; linewidth_data = 2.0; alpha = 0.3; 
mach = r'$\mathcal{M}$'
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)
#sharing x
ax2.sharex(ax1); ax3.sharex(ax1)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=0.05)

Re_Ran = []; Re_Med = []; Re_Err_down = []; Re_Err_up = []
Pm_Ran = []; Pm_Med = []; Pm_Err_down = []; Pm_Err_up = []

for i, m in enumerate(M):
    k_nu_ = K_nu[i]; k_nu_err_ = K_nu_err[i]; k_eta_ = K_eta[i]; k_eta_err_ = K_eta_err[i]
    p_val = get_p_val(m)

    Re_med, Re_err_down, Re_err_up, Re_ran = Re_ng_error(k_nu_, k_nu_err_, *cons(m,'nu'), p_val)
    Pm_med, Pm_err_down, Pm_err_up, Pm_ran = Pm_ng_error(k_nu_,k_nu_err_,k_eta_,k_eta_err_,*cons(m,'eta'),p_val)

    Re_Med.append(Re_med); Re_Err_down.append(Re_err_down); Re_Err_up.append(Re_err_down); Re_Ran.append(Re_ran)
    Pm_Med.append(Pm_med); Pm_Err_down.append(Pm_err_down); Pm_Err_up.append(Pm_err_up); Pm_Ran.append(Pm_ran)

Rm_Ran = [(Pm_Ran[i]*Re_Ran[i]).astype(float) for i in range(len(Pm_Ran))]
Rm_Med, Rm_Err_down, Rm_Err_up = Rm_ng_error(Rm_Ran)

Re_asymm_err = np.array(list(zip(Re_Err_down, Re_Err_up))).T
Pm_asymm_err = np.array(list(zip(Pm_Err_down, Pm_Err_up))).T
Rm_asymm_err = np.array(list(zip(Rm_Err_down, Rm_Err_up))).T

ax1.plot(M, Re_Med,'-',linewidth=linewidth,color='black')
add_errorbar(ax1, Re_Med, Re_asymm_err, M)
ax2.plot(M, Pm_Med,'-',linewidth=linewidth,color='black')
add_errorbar(ax2, Pm_Med, Pm_asymm_err, M)
ax3.plot(M, Rm_Med,'-',linewidth=linewidth,color='black')
add_errorbar(ax3, Rm_Med, Rm_asymm_err, M)
ax1.axvline(x=1,ymax=8e3,linestyle='dashed',color='k')
ax2.axvline(x=1,ymax=8,linestyle='dashed',color='k')
ax3.axvline(x=1,ymax=87e3,linestyle='dashed',color='k')
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,1))
ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,1))
ax1.set_ylabel(r'Re'); ax2.set_ylabel(r'Pm'); ax3.set_ylabel(r'Rm'); ax3.set_xlabel(r'$\mathcal{M}$')
ax1.set_ylim([1250,18e3]); ax2.set_ylim([-0.5,8]); ax3.set_ylim([-3e3,86e3])
fig = plt.gcf(); fig.suptitle('$N = $'+f' {N_res}', fontsize=20, y = 0.9)
#ax2.set_yscale('log'); ax4.set_yscale('log')
ax1.set_xscale('log'); ax2.set_xscale('log'); ax3.set_xscale('log')
filename = home_directory+folder+'/'+'RvsM.pdf'
plt.savefig(filename,format = "pdf")
plt.close()

# ************************************************* Table ******************************************************
k_nu_text = text_1(k_nu,k_nu_er,3); k_nu_act_text = text_1(k_nu_act,k_nu_er_act,3)
k_eta_text = text_1(k_eta,k_eta_er,3); k_eta_act_text = text_1(k_eta_act,k_eta_er_act,3)
P_eta_text = text_1(P_eta,P_eta_err,2); P_mag_text = text_1(P_mag,P_mag_err,3); P_bn_text = text_1(P_bn,P_bn_err,2); k_bn_text = text_1(k_bn,k_bn_er,3)
Re_text, Pm_text, Rm_text = text_2(Re_Med,Re_Err_down,Re_Err_up), text_2(Pm_Med,Pm_Err_down,Pm_Err_up), text_2(Rm_Med,Rm_Err_down,Rm_Err_up)

datafile_path = home_directory+folder+'/sim_table_mach.txt'
data = np.column_stack([['Mach']+M, P_kin, [P_bn[0]]+P_bn_text, P_nu, [k_bn[0]]+k_bn_text, [k_nu[0]]+k_nu_text, [P_mag[0]]+P_mag_text, [P_eta[0]]+P_eta_text, 
        [k_eta[0]]+k_eta_text, [k_nu_act[0]]+k_nu_act_text, [k_eta_act[0]]+k_eta_act_text, ['Re']+Re_text, ['Pm']+Pm_text, ['Rm']+Rm_text])
np.savetxt(datafile_path , data, delimiter = '\t',fmt = '%s')

quit()
# ************************************************* Compensated ******************************************************
mach_compare = [2.5, 1.25, 0.8]
plt.rcParams["figure.figsize"] = [10, 7]
plt.rcParams.update({'font.size': 7})
location = 'downer right'; fontsize = 5.5; alpha = 0.05
linewidth = 0.8; linewidth_data = 1.5
ax1 = plt.subplot()

for m in mach_compare:
    ind = M.index(m); c = color[ind]

    #Axis 1
    k = k_kin[ind]; y = E_tot_kin[ind]; y_fit = y_kin[ind]
    y_down = E_tot_err_down_kin[ind]; y_up = E_tot_err_up_kin[ind]
    index = Index(k,N_res); x = k

    ax1.plot(x, y*x**2,'-', color=c, linewidth=linewidth_data, label=mach+'$ =$'+' '+str(m))
    ax1.fill_between(x, y_down*x**2, y_up*x**2, color=c, alpha=alpha)
    ax1.plot(x[index], y_fit*x[index]**2,'-',linewidth=linewidth,color='black')
    #ax1.set_ylim([1e-13,1e2])
    ax1.legend(bbox_to_anchor = coord, labelspacing=labelspacing)
    ax1.axvline(x = K_nu[ind], ymax=0.035, color=c, linewidth = 3.0)

ylab = r'$P_{\rm '+'kin'+'}$'+'$k^2$'; ax1.set_ylabel(ylab); ax1.set_xscale('log'); ax1.set_yscale('log')
knu_label = r'$k_{\nu}$';ax1.set_xlabel(xlab)
ax1.text(0.62, 0.05, knu_label, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,fontsize=20,color='black')
filename = home_directory+folder+'/'+'spec_mach_comp.pdf'
plt.savefig(filename,format = "pdf")
plt.close()