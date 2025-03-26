import numpy as np
import cfpack as cfp
from cfpack import matplotlibrc, stop
np.seterr(divide='ignore', invalid='ignore')
import os
from vsN_funcs import *
home_directory = os.path.expanduser( '~' )+'/'

folder = home_directory+'/Decomp_spec_results/'
Dir = ['lon','tra']
Mach = 10

def file_access():

    def get_path(dire):
        textname = 'k_nu_vs_N_'+dire
        Path = folder+textname+'.txt'
        file = open(Path).readlines()
        return file

    col_num = 11
    par_len = col_num-3
    Params = [[] for _ in range(par_len)]
    k_act, k_act_err, N_res = [],[],[]

    for j, dire in enumerate(Dir):
        file = get_path(dire)
        k_act += get_col(file, col_num-1, 4+j)
        k_act_err += get_col(file, col_num, 4+j)
        N_res += get_col(file, 1, 4+j)
        for i, param in enumerate(Params):
            Params[i] = param + get_col(file, i+2, 4+j)

    return N_res, Params, k_act, k_act_err

N_res, Params_kin, k_nu_act, k_nu_act_err = file_access()
Re = ['Re']

for i, dire in enumerate(Dir):
    if i: start = N_len+1
    else: start = 1
    end = start + N_len

    re, re_err_up, re_err_down, _ = Re_ng_error(k_nu_act[start:end], k_nu_act_err[start:end], *cons(Mach,'nu'), get_p_val(Mach))
    Re = Re + text_2(re,re_err_up,re_err_down)

Tab_list_kin = make_tab(Params_kin)
k_nu_act = [k_nu_act[0]] + text_1(k_nu_act[1:],k_nu_act_err[1:],4)

datafile_path = folder+'sim_sep_table.txt'
data = np.column_stack([N_res,*Tab_list_kin,k_nu_act,Re])
np.savetxt(datafile_path , data, delimiter = '\t',fmt = '%s')