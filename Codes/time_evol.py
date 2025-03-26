import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cfpack as cfp
import numpy as np
from lmfit import Model, Parameters
import os
from statistics import stdev, mean
home_directory = os.path.expanduser( '~' )+'/'
folder = home_directory+'Simu_results'+'/'

def data_ext(N_res,Mach):
    if Mach<1: m = '0p'+ str(int(10*Mach))
    else: m = str(int(Mach))
    folder = 'M'+m+'N'+str(N_res).zfill(4)
    access = '/Turb.dat_cleaned'
    #linear regime
    data = cfp.read_ascii('/scratch/pawsey0810/cfederrath/turb_diss/'+folder+access)

    time = data['01_time'].value[1:]
    t_turn = 1/(2*Mach)
    time = time/t_turn
    Mach_No = data['#14_rms_velocity'].value[1:]
    E_mag = data['#12_E_magnetic'].value[1:]
    E_kin = data['#10_E_kinetic'].value[1:]
    Ene = E_mag/E_kin
    if Mach<1:index = (time>=1)
    else: index = (time>0)
    return time, Mach_No, Ene, index

#linear
def log_rate(t,T,E_0):
    E = E_0*np.exp(T*(t-4))
    return np.log10(E)

params = Parameters()
params.add('E_0', min=0, max=1e-5)
params.add('T', min=0)
model = Model(log_rate)

def Index(t,E,m):
    if m<1: ind = ((t>=4) & (E<0.8e-3))
    if m>1: ind = ((t>=4) & (E<0.5e-3))
    return ind

color = ['blue','orange','magenta','green','red','purple']
M = [0.1,10]
N_res = [2304,1152,576,288,144]
filename = 'time_evol'
#Plotting
fig = plt.figure(figsize = (20,12))
plt.rcParams.update({'font.size': 22})
location = 'lower right'; fontsize = 22; linewidth = 1.0

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

M_label = r'$\mathcal{M}$'
Ax_m = [ax1,ax2]; Ax_e = [ax3,ax4]; N_len = len(N_res); Tau = []; Tau_err = []

for m in M:
    if m<1: 
        ax_m = Ax_m[0]; ax_e = Ax_e[0]
        ax_m.set_ylabel(M_label)
        ax_e.set_ylabel(r'$E_{\mathrm{mag}}/E_{\mathrm{kin}}$')
        init = [[1e-10,5],[1e-11,3],[5e-12,2],[7e-12,1],[5e-11,1],[1e-10,0.7]]
    if m>1:
        ax_m = Ax_m[1]; ax_e = Ax_e[1]
        init = [[5e-8,0.7],[1e-8,0.7],[1e-8,0.6],[1e-8,0.5],[1e-8,0.4],[5e-8,0.5]]

    ax_e.set_xlabel(r'$t/t_{\mathrm{turb}}$')
    ax_e.axvline(x=4,ymin=1e-13,ymax=1e2,linestyle='dashed',color='k')
    ax_e.axhline(y=1e-3,xmin=-1,xmax=21,linestyle='dashed',color='k')
    ax_m.axvline(x=4,ymin=1e-2,ymax=1e2,linestyle='dashed',color='k')
    for i in np.arange(N_len):
        n = N_res[i]; c = color[i]
        time, Mach_No, Ene, ind = data_ext(n,m)
        ax_m.plot(time,Mach_No,color=c,linewidth=linewidth)
        ax_e.plot(time[ind],Ene[ind],color=c,label='$N =$'+' '+str(n))

        index = Index(time,Ene,m)
        E = Ene[index]; T = time[index]
        params['E_0'].value = init[i][0]
        params['T'].value = init[i][1]
        fit = model.fit(np.log10(E),t=T,params=params)
        print(fit.fit_report())
        y = log_rate(t=T,E_0=fit.values['E_0'],T=fit.values['T'])
        ax_e.plot(T,10**y,linestyle = '-.',color='black')
        #Saving data:
        #Store 'All free' parameter values
        for name, param in fit.params.items():
            if name == 'T': 
                Tau.append(param.value)
                Tau_err.append(param.stderr)
        
    time_ticks = [0,5,10,15,20]
    ax_e.set_xticks(time_ticks); ax_m.set_xticks(time_ticks)
    ax_e.set_ylim(1e-13,7e-1); ax_m.set_ylim(6e-2,13)
    ax_e.legend(fontsize=fontsize,loc=location)
    ax_e.set_yscale('log'); ax_m.set_yscale('log')
#Mach label
ax3.text(0.88, 0.49, M_label+' = 0.1', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes,fontsize=22,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))
ax4.text(0.88, 0.49, M_label+' = 10', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes,fontsize=22,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))

plt.savefig(folder+filename+'.pdf',format = "pdf")
plt.close()
print(f'Saved {filename}')
quit()

#Saving data:
N_res = np.array(['N_res']+N_res+N_res); Tau = np.array(['Tau']+Tau); Tau_err = np.array(['Tau_err']+Tau_err)
datafile_path = folder+filename+'.txt'
data = np.column_stack([N_res,Tau,Tau_err])
np.savetxt(datafile_path , data, delimiter = '\t',fmt = '%s')
print(f'{datafile_path}')

'''len_E = int(len(Ene[index])/5)
low = 0; up = 5*len_E'''

'''Tau_ = []; Err = []; Err_up = []; Err_dw = []
for i in range(1,6):
    Path = folder+f'time_evol_{i}.txt'
    file = open(Path).readlines()
    get_col = lambda col: (line.split()[col-1] for line in file[1:])
    Tau_.append(np.array(list(map(float,get_col(2)))))

for j in range(12):
    Val = []
    for i in range(len(Tau_)):
        Val.append(Tau_[i][j])
    err = stdev(Val)
    avg = mean(Val)
    err_ = 0
    for i in range(len(Val)):
        err_ = err_ + abs(Val[i]-avg)
    err_ = err_/len(Val)
    Err.append(err)'''