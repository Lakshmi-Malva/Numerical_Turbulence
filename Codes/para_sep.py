import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from cfpack import matplotlibrc, stop
from cfpack.defaults import *
import numpy as np
from lmfit import Model, Parameters
from vsN_funcs import *
np.seterr(divide='ignore', invalid='ignore')
home_directory = os.path.expanduser( '~' )+'/'

color = ['blue','orange','magenta','green','red','purple']
factor = 0.5*np.array([1,1e-1,1e-2,1e-3,1e-4]).astype(float)
N_res = [2304,1152,576,288,144]
Dir = ['lon','tra']
space = ['','','','']
spectype = 'kin'
spec = 'aver_spect_vels.dat'
Mach = 10; p_val = get_p_val(Mach); cons_nu = cons(Mach,'nu')[0]

def file_access(N_res, spectype, spec, dire):

    m = str(Mach)
    folder = 'M'+m+'N'+str(N_res).zfill(4)
    sub_folder = spectype+'_'+m+'_result'
    PATH = '/scratch/pawsey0810/cfederrath/turb_diss/'+folder+'/spectra/'    
    file = open(PATH+spec).readlines()
    get_col = lambda col: (line.split()[col-1] for line in file[6:])
    k = np.array(list(map(float,get_col(1))))

    #Longitudinal
    if dire == 'lon':
        log_E_tot = np.array(list(map(float,get_col(2))))
        log_E_tot_err = np.array(list(map(float,get_col(3))))

    #Transverse
    if dire == 'tra':
        log_E_tot = np.array(list(map(float,get_col(4))))
        log_E_tot_err = np.array(list(map(float,get_col(5))))

    E_tot = 10**log_E_tot 
    E_tot_err_up = 10**(log_E_tot+log_E_tot_err)
    E_tot_err_down = 10**(log_E_tot-log_E_tot_err)
    E_tot_err = (E_tot_err_down+E_tot_err_up)/2

    return k, E_tot, E_tot_err_up, E_tot_err_down, E_tot_err

def kin_est(N_res,dire,A,A_min,A_max,p_bn_val,k_bn_value,k_bn_min,k_bn_max,k_nu_value,k_nu_max):
    p_k = -2
    if dire == 'lon': p_nu = 1
    if dire == 'tra': p_nu = 0.7

    params = Parameters()
    params.add('A',value = A, min = A_min, max = A_max)
    params.add('p_kin',value = p_k)
    params['p_kin'].vary = False
    params.add('p_bn',value = p_bn_val,min=0, max=1)
    params.add('k_bn',value = k_bn_value,min=k_bn_min,max=k_bn_max)
    params.add('k_nu',value = k_nu_value, max=k_nu_max)
    params.add('p_nu', value = p_nu)
    params['p_nu'].vary = False

    #Accessing files and data
    k, E_tot, E_tot_err_up, E_tot_err_down, E_tot_err = file_access(N_res, spectype, spec, dire)

    #Fitting
    index = Index(k,N_res)

    weights = 1/E_tot_err[index]
    mod = Model(P_kin_fit_eq)
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
    print(f'k_nu_actual_{dire} = {k_nu_actual}')

    return k, E_tot, E_tot_err_up, E_tot_err_down, y, k_nu_value, k_nu_err, k_nu_actual, k_nu_err_act, k_bn_value, k_bn_err, p_kin, p_bn, p_bn_err, p_nu

#order = A,A_max,A_min, p_bn_val, k_bn_value,k_bn_min,k_bn_max, k_nu_value,k_nu_max
def Init(dire):
    if dire == 'lon':
        init = np.array([[3e-3,1,1e-4,1e-11,90,None,None,130,None],[9e-3,1e1,1e-4,1e-11,50,35,65,68,None],[3e-2,1e2,1e-5,2e-13,25,20,30,34,40],[9e-2,1e2,1e-5,2e-10,8,None,None,13,None],[2.7e-1,1e2,1e-5,3e-11,7,None,None,8,None]])
    if dire == 'tra':
        init = np.array([[9e-2,1,1e-4,1e-10,25,None,None,30,None],[3e-1,1e1,1e-4,1e-10,17,10,40,18,None],[0.9,1,1e-4, 1e-10, 10,None,None, 8,None],[3,1e2,1e-4,2e-10,6,None,None,4,None],[9,1e2,1e-4,3e-11,3,1,5,2,None]])
    return init

N_re = np.array(space+['N_res']+N_res)
k_lon = []; E_lon = []; E_lon_err_up = []; E_lon_err_down = []; Y_lon = []
k_tra = []; E_tra = []; E_tra_err_up = []; E_tra_err_down = []; Y_tra = []
K_nu = []
folder = 'Decomp_spec_results'

for dire in Dir:
    k_nu = space+['k_nu']; k_nu_er = space+['k_nu_error']; k_nu_act = space+['k_nu_act']
    k_nu_er_act = space+['k_nu_act_error']; k_bn = space+['k_bn']; k_bn_er = space+['k_bn_error']
    P_kin = space+['p_kin']; P_bn = space+['p_bn']; P_bn_err = space+['p_bn_error']; P_nu = space+['p_nu']

    for i in np.arange(len(N_res)):
        n = N_res[i]; init = Init(dire)[i]

        A = init[0]; A_min = init[2]; A_max = init[1]; p_bn_val = init[3]
        k_bn_value = init[4]; k_bn_min = init[5]; k_bn_max = init[6]; k_nu_value = init[7]; k_nu_max = init[8]
        k, E_tot, E_tot_err_up, E_tot_err_down, y, k_nu_value, k_nu_err, k_nu_actual, k_nu_err_act, k_bn_value, k_bn_err, p_kin, p_bn, p_bn_err, p_nu = kin_est(n,dire,A,A_min,A_max,p_bn_val,k_bn_value,k_bn_min,k_bn_max,k_nu_value,k_nu_max)

        if dire == 'lon': k_lon.append(k); E_lon.append(E_tot); E_lon_err_down.append(E_tot_err_down); E_lon_err_up.append(E_tot_err_up); Y_lon.append(y)
        if dire == 'tra': k_tra.append(k); E_tra.append(E_tot); E_tra_err_down.append(E_tot_err_down); E_tra_err_up.append(E_tot_err_up); Y_tra.append(y)
        
        k_nu.append(k_nu_value); k_nu_er.append(k_nu_err); k_nu_act.append(k_nu_actual); k_nu_er_act.append(k_nu_err_act); k_bn.append(k_bn_value)
        k_bn_er.append(k_bn_err); P_kin.append(p_kin); P_bn_err.append(p_bn_err); P_bn.append(p_bn); P_nu.append(p_nu)
        K_nu.append(k_nu_actual)

    filename = 'k_nu_vs_N_'+dire+'.txt'               
    datafile_path = home_directory+folder+'/'+filename
    data_kin = np.column_stack([N_re, P_kin, P_bn, P_bn_err, P_nu, k_bn, k_bn_er, k_nu, k_nu_er, k_nu_act, k_nu_er_act])
    np.savetxt(datafile_path , data_kin, delimiter = '\t',fmt = '%s')
    print(f'Saved{filename}')

#Plotting
'''fig = plt.figure(figsize = (20,12))
plt.rcParams.update({'font.size': 10})
location = 'upper right'; fontsize = 8
coord = (0.75, 0.91); labelspacing = 0.1
linewidth = 1.0; linewidth_data = 2.0; alpha = 0.3
M = r'$\mathcal{M}$'

ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
#sharing x
plt.setp(ax1.get_xticklabels(), visible=False)
plt.subplots_adjust(wspace=0.025, hspace=0.05)
filename = 'spectra_sep_comp.pdf'

K_nu = np.array(K_nu); N_len = len(N_res)
for i in np.arange(N_len):
    n = N_res[i]; c = color[i]
    for dire in Dir:
        ylab = r'$P_{\rm '+dire+'}$'+'$k^2$'
        if dire == 'lon': 
            k = k_lon[i]; E_tot = E_lon[i]; Y = Y_lon[i]
            E_tot_err_down = E_lon_err_down[i]
            E_tot_err_up = E_lon_err_up[i]
            ax = ax1; k_nu_mark = K_nu[i]
        
        if dire == 'tra': 
            k = k_tra[i]; E_tot = E_tra[i]; Y = Y_tra[i]
            E_tot_err_down = E_tra_err_down[i]; E_tot_err_up = E_tra_err_up[i]
            ax = ax2; ax.set_xlabel(xlab); k_nu_mark = K_nu[i+N_len]

        index = Index(k,n)
        y = factor[i]*E_tot*k**2; x = k
        y_fit = factor[i]*Y*k[index]**2
        y_down = factor[i]*E_tot_err_down*k**2
        y_up = factor[i]*E_tot_err_up*k**2

        ax.plot(x, y,'-', color=c, linewidth=linewidth_data, label='$N =$'+' '+str(n))
        ax.fill_between(x, y_down, y_up, color=c, alpha=alpha)
        ax.plot(x[index], y_fit,'-',linewidth=linewidth,color='black')
        ax.axvline(x = k_nu_mark, ymax=0.05, color=c, linewidth = 3.0)
        #ax.text(k_nu_mark, 1/16, k_label, horizontalalignment='center', verticalalignment='center', transform=trans1,fontsize=fontsize, color=c)
        ax.set_ylabel(ylab)
        ax.set_xscale('log')
        ax.set_yscale('log')
        if dire == 'lon': ax.legend(fontsize=fontsize,loc=location)

plt.savefig(home_directory+folder+'/'+filename,format = "pdf")
print(f'Saved {filename}')'''

fig = plt.figure(figsize = (20,12))
plt.rcParams.update({'font.size': 20})
fontsize = 15; coord = (0.75, 0.88)
location = 'upper right'
coord = (0.75, 0.91); labelspacing = 0.1
linewidth = 1.0; linewidth_data = 2.0; alpha = 0.3
M = r'$\mathcal{M}$'

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223, sharex=ax1)
ax4 = plt.subplot(224, sharex=ax2)
#sharing x
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
trans1 = ax1.get_xaxis_transform()
trans2 = ax3.get_xaxis_transform()
k_label = knu_label

plt.subplots_adjust(wspace=0.12, hspace=0.05)
filename = 'Result_seperate.pdf'

K_nu = np.array(K_nu); N_len = len(N_res)
for i in np.arange(len(N_res)):
    n = N_res[i]; c = color[i]
    for dire in Dir:
        if dire == 'lon': 
            k = k_lon[i]; E_tot = E_lon[i]; Y = Y_lon[i]
            E_tot_err_down = E_lon_err_down[i]; E_tot_err_up = E_lon_err_up[i]
            ax = ax1; k_nu_mark = K_nu[i]; trans = trans1
            ylab = r'$P_{\rm '+'kin'+r'_{\parallel}}$'
        
        if dire == 'tra': 
            k = k_tra[i]; E_tot = E_tra[i]; Y = Y_tra[i]
            E_tot_err_down = E_tra_err_down[i]; E_tot_err_up = E_tra_err_up[i]
            ax = ax3; ax.set_xlabel(xlab); k_nu_mark = K_nu[i+N_len]; trans = trans2
            ylab = r'$P_{\rm '+'kin'+r'_{\perp}}$'

        index = Index(k,n)
        y = factor[i]*E_tot; x = k
        y_fit = factor[i]*Y
        y_down = factor[i]*E_tot_err_down; y_up = factor[i]*E_tot_err_up
        
        ax.plot(x, y,'-', color=c, linewidth=linewidth_data, label='$N =$'+' '+str(n))
        ax.fill_between(x, y_down, y_up, color=c, alpha=alpha)
        ax.plot(x[index], y_fit,'-',linewidth=linewidth,color='black')
        ax.axvline(x = k_nu_mark, ymax=0.035, color=c, linewidth = 3.0)
        ax.text(k_nu_mark, 1/16, k_label, horizontalalignment='center', verticalalignment='center', transform=trans,fontsize=fontsize, color=c)
        ax.set_ylabel(ylab)
        ax.set_xscale('log')
        ax.set_yscale('log')
        if dire == 'lon': ax.legend(fontsize=fontsize,bbox_to_anchor = coord)
        if dire == 'tra': ax.legend(fontsize=fontsize,loc=location)

ax1.text(0.87, 0.9, M+'= 10', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,fontsize=15,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))

# ************************************************* k vs N ******************************************************

Textname = []
K_nu = []; K_nu_err = []; N_Re_value = []; N_Re_err = []; p_Re_value = []; p_Re_err = []

def file_access(dire):
    textname = 'k_nu_vs_N_'+dire
    Path = home_directory+folder+'/'+textname+'.txt'
    file = open(Path).readlines()
    get_col = lambda col: (line.split()[col-1] for line in file[5:11])
    return textname, get_col

for dire in Dir:
    textname, get_col = file_access(dire)

    N_res = np.array(list(map(float,get_col(1)))) 
    k_nu = np.array(list(map(float,get_col(10))))
    k_nu_err = np.array(list(map(float,get_col(11))))

    K_nu.append(k_nu); K_nu_err.append(k_nu_err); Textname.append(textname)

#Define:
elinewidth=0.85; ecolor='k'; capsize=3

# ============================================== K_NU ===============================================
#PLOTS:
location = 'lower right'

model = log_k_nu_of_N; Ax = [ax2,ax4]; order = [3,0,2,1]
para, mod = add_params_fit(model = model, p_val = p_val, N_res = N_res)

for i in range(len(Dir)):
    k_nu = K_nu[i]; k_nu_err = K_nu_err[i]; weights = k_nu/(2*k_nu_err); ax = Ax[i]; dire = Dir[i]

    #Data & Errorbar:
    add_errorbar(ax = ax, y = k_nu, y_err = k_nu_err, N_res = N_res)

    #Theory
    params = [3.0,p_val,cons_nu]
    add_params_theory(N_res = N_res, ax = ax, model = model, p_val = p_val, values = params, color = 'k')

    #print(' ********************* ALL FREE ********************** ')

    n_re_mach = 3.
    my_params = {
        'N_Re': [n_re_mach*0.5, n_re_mach, n_re_mach*5],
        'p_Re': [p_val*0.8, p_val, p_val*2],
        'cons_nu': [None, cons_nu, None],
        'p_val': [None, p_val, None]
    }
    
    params_bfre = perform_fit(model, N_res, k_nu, k_nu_err, my_params)
    #Params_bfre[Dir[i]] = params_bfre
    
    y, label = plot_creat_ret(model, params_bfre, N_res)
    ax.plot(N_res,y,'-',color='orange',label= label)

    #Saving data:
    with open(folder+'/'+Textname[i]+'.txt', 'r') as file: data = file.readlines()
    data[0] = f"N_Re = {params_bfre['N_Re'][0]}\n"
    data[1] = f"N_Re_error = {params_bfre['N_Re'][1]}+{params_bfre['N_Re'][2]}\n"
    data[2] = f"p_Re = {params_bfre['p_Re'][0]}\n"
    data[3] = f"p_Re_error = {params_bfre['p_Re'][1]}+{params_bfre['p_Re'][2]}\n"
    with open(folder+'/'+Textname[i]+'.txt', 'w') as file: file.writelines(data)

    #print(' ********************* N_Re FREE ********************* ')

    my_params['p_Re'] = [None, p_val, None]
    params_nfre = perform_fit(model, N_res, k_nu, k_nu_err, my_params)
    y, label = plot_creat_ret(model, params_nfre, N_res)
    ax.plot(N_res,y,'--',color='blue',label=label)

    #Design:
    ax.set_xscale('log')
    ax.set_yscale('log')
    if dire == 'lon': ylab = r'$k_{\nu_{\parallel}}$'
    if dire == 'tra': ylab = r'$k_{\nu_{\perp}}$'
    ax.set_ylabel(ylab)
    if ax == ax4: 
        ax.set_xlabel('$N$')
        handles, labels = fig.get_axes()[3].get_legend_handles_labels()
    else: handles, labels = fig.get_axes()[1].get_legend_handles_labels()
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=fontsize, loc=location)
    ax.set_xlim([np.min(N_res)-20,np.max(N_res)+500])

#Mach label
M = r'$\mathcal{M}$'
ax2.text(0.09, 0.9, M+'= 10', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes,fontsize=15,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))
plt.savefig(home_directory+folder+'/'+filename,format = "pdf")
plt.close()
print(f'Saved {filename}')
