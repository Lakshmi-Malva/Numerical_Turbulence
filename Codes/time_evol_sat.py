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
    #saturation regime
    data = cfp.read_ascii('/scratch/pawsey0810/cfederrath/turb_diss/'+folder+'/resim'+access)

    time = data['01_time'].value[1:]
    t_turn = 1/(2*Mach)
    time = time/t_turn
    Mach_No = data['#14_rms_velocity'].value[1:]
    E_mag = data['#12_E_magnetic'].value[1:]
    E_kin = data['#10_E_kinetic'].value[1:]
    Ene = E_mag/E_kin
    index = ((time>=10) & (time<=47))
    return time, Mach_No, Ene, index

def log_rate(t,E_0,a):
    E = E_0 + a*t
    return np.log10(E)

'''def log_rate(t,E_0):
    E = E_0 
    return np.log10(E)'''

params = Parameters()
params.add('E_0', min=1e-3,max=1)
params.add('a', value=0)
params['a'].vary = False
model = Model(log_rate)

def Index(t):
    ind = ((t>=42) & (t<=47))
    return ind

#color = ['blue','orange','magenta','green','red','purple']
color = ['grey']
M = [0.1,10]
#N_res = [2304,1152,576,288,144]
N_res = [576]
filename = 'time_evol_sat'
#Plotting
fig = plt.figure(figsize = (10,12))
plt.rcParams.update({'font.size': 20})
location = 'lower right'; fontsize = 15; linewidth = 1.0

ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex = ax1)
#sharing x
plt.setp(ax1.get_xticklabels(), visible=False)
plt.subplots_adjust(wspace=0.01, hspace=0.01)

M_label = r'$\mathcal{M}$'
Ax = [ax1,ax2]; N_len = len(N_res)
ax1.set_ylabel(M_label)
ax2.set_ylabel(r'$E_{\mathrm{mag}}/E_{\mathrm{kin}}$')
ax2.set_xlabel(r'$t/t_{\mathrm{turb}}$')
ax2.axhline(y=1e-3,xmin=-1,xmax=21,linestyle='dashed',color='k')
ax1.set_yscale('log'); ax2.set_yscale('log')

Tau = []; Tau_err = []
for m in M:
    for i in np.arange(N_len):
        n = N_res[i]; c = 'purple'
        time, Mach_No, Ene, ind = data_ext(n,m)
        if m<1:
            ax1.plot(time[ind],Mach_No[ind],color=c,linewidth=linewidth)
            ax2.plot(time[ind],Ene[ind],color=c,label=M_label+' = 0.1')
            #params = {'E_0':[1e-3,0.5,1]}
            params['E_0'].value = 0.5
        else:
            ax1.plot(time[ind],Mach_No[ind],'--',color=c,linewidth=linewidth)
            ax2.plot(time[ind],Ene[ind],'--',color=c,label=M_label+' = 10')
            #params = {'E_0':[1e-3,1e-2,1]}
            params['E_0'].value = 1e-2
        
        index = Index(time)
        E = Ene[index]; T = time[index]
        fit = model.fit(np.log10(E),t=T,params=params)
        #print(fit.fit_report())
        y = log_rate(t=T,E_0=fit.values['E_0'],a=fit.values['a'])
        ax2.plot(T,10**y,linestyle = 'dotted',color='black')

        #Saving data:
        #Store 'All free' parameter values
        for name, param in fit.params.items():
            if name == 'E_0': 
                Tau.append(param.value)
                Tau_err.append(param.stderr)
        
        '''fit = cfp.fit(log_rate,E,T,perr_method='statistical',params=params)
        for ip, pname in enumerate(fit.pnames):
            E_0 = fit.popt[ip]; Tau.append(E_0)
            E_0_err = (fit.perr[ip][0]+fit.perr[ip][1])/2
            Tau_err.append(E_0_err)
            break'''

ax2.set_ylim(1e-7,1.1); ax1.set_ylim(6e-2,13); ax2.set_xlim(9,48)
ax2.legend(fontsize=20,loc=location)
ax1.tick_params(axis="x",direction="in"); ax1.tick_params(axis="y",direction="in")
ax2.tick_params(axis="x",direction="in"); ax2.tick_params(axis="y",direction="in")
ax1.tick_params(axis="y",which='minor',direction="in"); ax2.tick_params(axis="y",which='minor',direction="in")

plt.savefig(folder+filename+'.pdf',format = "pdf",bbox_inches='tight')
plt.close()
print(f'Saved {filename}')

#Saving data:
N_res = np.array(['Mach']+M); Tau = np.array(['E_0']+Tau); Tau_err = np.array(['E_0_err']+Tau_err)
datafile_path = folder+filename+'.txt'
data = np.column_stack([N_res,Tau,Tau_err])
np.savetxt(datafile_path , data, delimiter = '\t',fmt = '%s')
print(f'{datafile_path}')
