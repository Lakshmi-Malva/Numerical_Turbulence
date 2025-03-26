import numpy as np
import cfpack as cfp
from cfpack import matplotlibrc, stop
np.seterr(divide='ignore', invalid='ignore')
import os
from vsN_funcs import *
home_directory = os.path.expanduser( '~' )+'/'

folder = 'Simu_results'

def file_access(spect):

    if spect == 'kin': const = 'nu'; col_num = 11
    if spect == 'mag': const = 'eta'; col_num = 9

    textname = 'k_'+const+'_vs_mach_576'
    Path = home_directory+folder+'/'+textname+'.txt'
    file = open(Path).readlines()
    
    par_len = col_num-3
    Params = [[] for _ in range(par_len)]
    k_act, k_act_err, Mach = [],[],[]

    k_act += get_col(file, col_num-1, 4)
    k_act_err += get_col(file, col_num, 4)
    Mach += get_col(file, 1, 4)
    for i, param in enumerate(Params):
        Params[i] = param + get_col(file, i+2, 4)

    return Mach, Params, k_act, k_act_err

Mach, Params_kin, k_nu_act, k_nu_act_err = file_access('kin')
_, Params_mag, k_eta_act, k_eta_act_err = file_access('mag')
Re = ['Re']; Pm = ['Pm']; Rm = ['Rm'] 
Re_Med = []; Re_Err_down = []; Re_Err_up = []
Pm_Med = []; Pm_Err_down = []; Pm_Err_up = []

for i, m in enumerate(Mach[1:]):

    re, re_err_up, re_err_down, re_ran = Re_ng_error([k_nu_act[i+1]], [k_nu_act_err[i+1]], *cons(m,'nu'), get_p_val(m))
    pm, pm_err_up, pm_err_down, pm_ran = Pm_ng_error([k_nu_act[i+1]], [k_nu_act_err[i+1]], [k_eta_act[i+1]], [k_eta_act_err[i+1]], *cons(m,'eta'), get_p_val(m))
    rm_ran = pm_ran*re_ran
    rm, rm_err_up, rm_err_down = Rm_ng_error(rm_ran.astype(float))

    Re = Re + text_2(re,re_err_up,re_err_down)
    Pm = Pm + text_2(pm,pm_err_up,pm_err_down)
    Rm = Rm + text_2(rm,rm_err_up,rm_err_down)

Tab_list_kin = make_tab(Params_kin)
Tab_list_mag = make_tab(Params_mag)

k_nu_act = [k_nu_act[0]] + text_1(k_nu_act[1:],k_nu_act_err[1:],3)
k_eta_act = [k_eta_act[0]] + text_1(k_eta_act[1:],k_eta_act_err[1:],3)

datafile_path = home_directory+folder+'/'+'sim_table_mach.txt'
data = np.column_stack([Mach,*Tab_list_kin,*Tab_list_mag,k_nu_act,k_eta_act,Re,Pm,Rm])
np.savetxt(datafile_path , data, delimiter = '\t',fmt = '%s')