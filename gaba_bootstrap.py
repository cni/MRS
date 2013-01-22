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
w_data, w_supp_data = coil_combine(data)

CIs = bootstrap.ci(w_supp_data, statfunction=get_single_spectra, alpha=0.05, n_samples=10000, method='bca')
