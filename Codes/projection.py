import flashplotlib as fpl
from cfpack import matplotlibrc
import numpy as np
import os

def slices(Mach,N_res,data,time_stamp_no): 
    if Mach<1: 
        m = '0p'+ str(int(10*Mach))
        data_mag = (3.54491e-11**2)/(8*np.pi)
    else: 
        m = str(int(Mach))
        data_mag = (3.54491e-09**2)/(8*np.pi)
    folder = 'M'+m+'N'+str(N_res).zfill(4)
    time_stamp = str(int(10*time_stamp_no)).zfill(4)
    Path_in = '/scratch/pawsey0810/cfederrath/turb_diss/'+folder
    Path_out = '/home/laksh253/slices'
    file_in = f'Turb_hdf5_plt_cnt_{time_stamp}'
    file_out = f'{folder}_{data}_{time_stamp}'
    t_turn = 1/(2*Mach)
    
    if data == 'dens': 
        map_label = 'Density (projection length $z = L$)'
    elif data == 'ekin': 
        map_label = '$E_\mathrm{kin}$ (projection length $z = L$)'
    elif data == 'emag': 
        map_label = '$E_\mathrm{mag}/E_{\mathrm{mag},\mathrm{0}}$ (projection length $z = L$)'

    #os.system('python3 python/interactive_job.py -n=16')
    # parse arguments
    args = fpl.parse_args()
    args.datasetname = f'{data}'
    args.outtype = ['pdf']
    args.outname = f'{file_out}'
    args.outdir = f'{Path_out}'
    args.time_unit = ''
    args.axes_unit = ['','','']
    args.cmap_label = f'{map_label}'
    args.time_scale = t_turn 
    args.fontsize = 1.5
    args.verbose = 2
    #if N_res == 2304: args.ncpu = 96

    if data == 'emag': args.data_transform = f'q/{data_mag}'
    else: args.data_transform = 'q' 
    if data == 'dens' and N==2304: 
        if Mach<1: args.data_transform = f'q/{9.697256e-01}'
        else: args.data_transform = f'q/{9.9278277158737183e-01}'
    else: args.data_transform = 'q' 
    
    args.Nres = f'{N_res}'
    if N_res == 576 or N_res == 144:
        args.colorbar = 0
    if N_res == 2304:
        args.colorbar = 1
    if data == 'dens':
        if Mach<1:
            args.vmax = 1.004
            args.vmin = 0.996
        if Mach>1:
            args.vmax = 4
            args.vmin = 1e-1
    if data == 'ekin':
        args.stream = True
        args.stream_color = 'white'
        args.stream_var = 'vel'
        if Mach<1:
            args.vmax = 1e-2
            args.vmin = 1e-3
        if Mach>1:
            args.vmax = 3e2
            args.vmin = 5
    if data == 'emag':
        args.stream = True
        args.stream_color = 'white'
        if Mach<1:
            if N_res == 2304: args.stream_color = 'purple'
            args.vmax = 1e19
            args.vmin = 1e11
        if Mach>1:
            args.vmax = 1e16
            args.vmin = 1e12
    if N_res == 576 or N_res == 2304:
        if data == 'ekin' or 'dens':
            args.axes_format = ['','','']
            args.axes_label = ['','','']
        if data == 'emag': 
            args.axes_format = [None,'','']
            args.axes_label = ['$x/L$','','']
    if N_res == 144:
        if data == 'ekin' or 'dens':
            args.axes_format = ['',None,'']
            args.axes_label = ['','$y/L$','']
        if data == 'emag': 
            args.axes_format = [None,None,'']
            args.axes_label = ['$x/L$','$y/L$','']
    if data == 'ekin': args.cmap = 'viridis'
    if data == 'emag': args.cmap = 'plasma'
    # process file
    fpl.process_file(f'{Path_in}/{file_in}', args)
    return

M = [0.1]
Data = ['dens']
#Data = ['emag','dens','ekin']
#N_res = [144,576]
#N_res = [2304]
N_res = [2304,144,576]
Time_stamp = 10

for Mach in M:
    for N in N_res:
        for data in Data:
            t = Time_stamp
            slices(Mach,N,data,t)

