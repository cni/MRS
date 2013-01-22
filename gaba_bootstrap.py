#! /usr/bin/python

import sys
import os

import scipy  
import scikits.bootstrap as bootstrap  

from MRS.quant import *
from MRS.analysis import get_spectra, get_single_spectra
import MRS.files as io

pfile = str(sys.argv[1])
data_dir = '/home/grace/spectro/data/'

# read file, convert to npy format 
data = io.get_data(os.path.join(data_dir, pfile))
data = data.squeeze() # get rid of singleton dimensions

#np.save(os.path.join(data_dir,pfile[:-2]), data) # save in .npy format

#amp_on, amp_off = reconstruct(pfile)
#quantify(pfile)

hdr = io.get_header(os.path.join(data_dir, pfile))
nTransients = (hdr['nframes']-4)*2
print 'number of transients: ' + str(nTransients) # which one is number of transients?

### bootstrap ###
# coil combine, get water signal
w_data, w_supp_data = coil_combine(data)
f_w, w_sig = get_single_spectra(w_data)

w_supp_data = w_supp_data.swapaxes(0,1) # swap axes so that the transients are axis = 0
print w_supp_data.shape

CIs = bootstrap.ci(w_supp_data, statfunction=get_single_spectra, alpha=0.05, n_samples=10, method='bca')  # should use 10000 samples, but here reduced to see if this is working

print CIs
#(UL,LL) = CIs # upper and lower limit

# normalize with water signal
#corrected_UL = normalize_water(w_sig, CIs)
