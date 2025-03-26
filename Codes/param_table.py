import numpy as np
import cfpack as cfp
from cfpack import matplotlibrc, stop
np.seterr(divide='ignore', invalid='ignore')
import os, random
from vsN_funcs import *
home_directory = os.path.expanduser( '~' )+'/'

folder = 'Simu_results'
Mach = [0.1,10]

def file_access(spect):

    if spect == 'kin': 
        const = 'nu'; col_num = 11
    if spect == 'mag': 
        const = 'eta'; col_num = 9

    def get_path(m):
        if m<1: M= '0p'+str(int(10*m))
        else: M = str(int(m))  
        #textname = 'k_'+const+'_vs_N_'+M
        textname = 'k_'+const+'_vs_N_'+M+'sat'
        Path = home_directory+folder+'/'+textname+'.txt'
        file = open(Path).readlines()
        return file

    par_len = col_num-3
    Params = [[] for _ in range(par_len)]
    k_act, k_act_err, N_res = [],[],[]

    for j, m in enumerate(Mach):
        file = get_path(m)
        k_act += get_col(file, col_num-1, 4+j)
        k_act_err += get_col(file, col_num, 4+j)
        N_res += get_col(file, 1, 4+j)
        for i, param in enumerate(Params):
            Params[i] = param + get_col(file, i+2, 4+j)
    
    return N_res, Params, k_act, k_act_err

#path = home_directory+folder+'/time_evol.txt'
path = home_directory+folder+'/time_evol_sat.txt'
file = open(path).readlines()
Tau = list(map(float,(line.split()[1] for line in file[1:])))
Tau_err = list(map(float,(line.split()[2] for line in file[1:])))

N_res, Params_kin, k_nu_act, k_nu_act_err = file_access('kin')
_, Params_mag, k_eta_act, k_eta_act_err = file_access('mag')
N_len = int(len(N_res[1:])/2)
Re = ['Re']; Pm = ['Pm']; Rm = ['Rm']

for i, m in enumerate(Mach):
    if i: start = N_len+1
    else: start = 1
    end = start + N_len

    #stop()
    print(k_nu_act[start:end])
    re, re_err_up, re_err_down, re_ran = Re_ng_error(k_nu_act[start:end], k_nu_act_err[start:end], *cons(m,'nu'), get_p_val(m))
    pm, pm_err_up, pm_err_down, pm_ran = Pm_ng_error(k_nu_act[start:end],k_nu_act_err[start:end],k_eta_act[start:end],k_eta_act_err[start:end],*cons(m,'eta'),get_p_val(m))
    rm_ran = pm_ran*re_ran
    rm, rm_err_up, rm_err_down = Rm_ng_error(rm_ran.astype(float))
    
    Re = Re + text_2(re,re_err_up,re_err_down)
    Pm = Pm + text_2(pm,pm_err_up,pm_err_down)
    Rm = Rm + text_2(rm,rm_err_up,rm_err_down)

Tab_list_kin = make_tab(Params_kin)
Tab_list_mag = make_tab(Params_mag)

k_nu_act = [k_nu_act[0]] + text_1(k_nu_act[1:],k_nu_act_err[1:],3)
k_eta_act = [k_eta_act[0]] + text_1(k_eta_act[1:],k_eta_act_err[1:],3)
Tau = ['Tau'] + text_1(Tau,Tau_err,2)

#datafile_path = home_directory+folder+'/'+'sim_table.txt'
datafile_path = home_directory+folder+'/'+'sim_table_sat.txt'
data = np.column_stack([N_res,Tau,*Tab_list_kin,*Tab_list_mag,k_nu_act,k_eta_act,Re,Pm,Rm])
np.savetxt(datafile_path , data, delimiter = '\t',fmt = '%s')
print(f'Saved {datafile_path}')