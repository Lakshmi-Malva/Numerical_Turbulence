import matplotlib, random
import matplotlib.pyplot as plt
import numpy as np
from cfpack import get_pdf, round

def add_errorbar(ax, y, y_err, x):
    line = ax.errorbar(x,y,yerr=2*y_err,ls='none',marker='o',mfc='white',mec='k',barsabove=True,elinewidth=0.85,ecolor='k',capsize=3)
    return line

#Ng_error:
n_ran = int(1e4); bins = 1000

def generate_random_gaussian_numbers(n, mu, sigma, seed=None):
    random.seed(seed) # set the random seed; if None, random uses the system time
    random_numbers = [random.gauss(mu, sigma) for _ in range(n)]
    return np.array(random_numbers)
    
def ng_error(c, log = False):
    # Now get some statistics of c
    # first get the PDF data and define the binsize
    if log: ret = get_pdf(np.log10(c), range=None, bins=bins)
    else: ret = get_pdf(c, range=None, bins=bins)
    pdf = ret.pdf; bin_centre = ret.bin_center
    binsize = bin_centre[1]-bin_centre[0] # this is the bin width
    loc = bin_centre - 0.5*binsize # this is the left edge of each bin location
    cdf = np.cumsum(pdf)*binsize # this is the cumulative distribution function

    # now compute the 16th and 84th percentile
    index = cdf >= 0.16
    if log: sixteenth = 10**(loc[index][0])
    else: sixteenth = loc[index][0]
    index = cdf >= 0.84
    if log: eightyfourth = 10**(loc[index][0])
    else: eightyfourth = loc[index][0]
    # compute the median
    index = cdf >= 0.5
    if log: median_c = 10**(loc[index][0])
    else: median_c = loc[index][0]

    # now pick your preferred best value; here let's pick the median
    err_low = median_c - sixteenth
    err_high = eightyfourth - median_c
    
    return median_c, err_low, err_high

def text_1(arr1,arr2,n):
    ans = []
    for i in range(1,len(arr1)):
        if arr1[i]<1e-8: arr1[i] = 0
        val = round(arr1[i],nfigs=n)
        err = round(arr2[i],nfigs=2)
        ans.append(f'${val}'+r'\!\pm\!'+f'{err}$')
    return ans

def text_2(arr1,arr2,arr3):
    i = 0
    while (arr1>10):
        arr1 = arr1/10
        i = i+1
    div = np.power(10,i)
    val = round(arr1,nfigs=2)
    err1 = round(arr2/div,nfigs=1)
    err2 = round(arr3/div,nfigs=1)
    if i!=0: ans = (f'${val}'+r'_{-'+f'{err1}'+'}'+r'^{+'+f'{err2}'+'}'+r'\!\times\! 10^{'+f'{i}'+'}$')
    else: ans = (f'${val}'+r'_{-'+f'{err1}'+'}'+r'^{+'+f'{err2}'+'}$')
    return ans