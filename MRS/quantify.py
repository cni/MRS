#!/usr/bin/python

"""

Soher 1996, "quantitation of automated single voxel proton MRS using cerebral
water as an internal reference"

""" 
import os
import sys

import numpy as np
import scipy
import scipy.optimize as opt
import scipy.integrate
import matplotlib.pyplot as plt
import sys
from scipy.ndimage.filters import maximum_filter1d
from pylab import *
import array
import scipy.linalg as la
import MRS.utils as ut
from MRS.peaks import * 

import nitime as nt
import nitime.timeseries as nts 
import nitime.analysis as nta
import nitime.viz as viz

import MRS.files as io
from MRS.analysis import coil_combine,normalize_water, get_spectra
from MRS.analysis_onoffsplit import two_echo_coil_combine
import MRS.utils as ut
from MRS.peaks import *

import array

sage = os.environ['SAGE_DATABASE']

# read file
exam = str(sys.argv[1])
#exam = 'SM3714/SM_LStr2'

# echo 1 - GABA suppressed
f1 = open(sage+'/'+exam+'/echo1/P11111.7_combine_pro.sdf', 'rb')
d1 = array.array('f')
d1.fromfile(f1, 1068)
# echo 2 - GABA non-suppressed 
f2 = open(sage+'/'+exam+'/echo2/P11111.7_combine_pro.sdf', 'rb')
d2 = array.array('f')
d2.fromfile(f2, 1068)

# take real numbers only 
d1real = d1[::2]
d2real = d2[::2]
# calculate difference
diff_tmp = d2
for i in range(len(diff_tmp)):
    diff_tmp[i] -= d1[i]
diff = diff_tmp[::2]

# number of points in data
npts = len(d1real)

xmin = 4.3 # x axis lower bound
xmax = -0.8 # x axis upper bound

# plot fig
xAxis = np.arange(xmin,xmax,(xmax-xmin)/npts)
figure(1)
plot(xAxis,d1real)
plot(xAxis,d2real)
plot(xAxis,diff)
xlim(xmin,xmax)

# detect peaks -- should really make a function that will search for peaks near known peaks
# returns in index not ppm
peaks1 = peakdetect(d1real)
peaks2 = peakdetect(d2real)
peaksDiff = peakdetect(diff)

# convert to ppm
ppmPeaks1 = array.array('f')
ppmPeaks2 = array.array('f')
ppmPeaksDiff = array.array('f')
for i in range(len(peaks1)):
	ppmPeaks1.append(peaks1[i]*(xmax-xmin)/npts+xmin)
for i in range(len(peaks2)):
	ppmPeaks2.append(peaks2[i]*(xmax-xmin)/npts+xmin)
for i in range(len(peaksDiff)):
	ppmPeaksDiff.append(peaksDiff[i]*(xmax-xmin)/npts+xmin)

# plot - should visually check!
for idx in range(len(peaks2)):
	plot(ppmPeaks2[idx],d2real[int(peaks2[idx])], 'co')
for idx in range(len(peaks1)):
	plot(ppmPeaks1[idx],d1real[int(peaks1[idx])], 'ro')
for idx in range(len(peaksDiff)):
	plot(ppmPeaksDiff[idx],diff[int(peaksDiff[idx])], 'mo')
plt.show()


# convolve with lorentzian
X1 = conv_lorentz(peaks1, d1real)
X2 = conv_lorentz(peaks2, d2real)
Xdiff = conv_lorentz(peaksDiff, diff)

# calculate the betas
res1 = np.dot(ols_matrix(X1.T),d1real) # dot multiple ordinary least squares matrix and Y
res2 = np.dot(ols_matrix(X2.T),d2real) # dot multiple ordinary least squares matrix and Y
resDiff = np.dot(ols_matrix(Xdiff.T),diff) # dot multiple ordinary least squares matrix and Y

# convert peak x axis units to ppm and print corresponding beta
print 'RESULTS:'
for i in range(len(peaks1)):
	ppm =  xmin - (peaks1[i]/npts * (xmin-xmax))
	print 'ppm: ' + str(ppm) + '; amp: ' + str(res1[0,i])

#plt.show()

# verify - multiply design matrix by betas, do you get something resembling raw data?

# calculate GABA peak from difference between echo 2 and echo 1

# reference compound peak
# Wang 2006 paper's reference is Creatine in unedited spectrum
# Creatine = 3.0ppm in echo2? why not take from GABA-suppressed instead of unedited
CrPeakIdx = peak_nearest(3.0, ppmPeaks2)
CrAmp = res2[0,CrPeakIdx]

# GABA = 3.0ppm in diff (GABA edited)
GABAPeakIdx = peak_nearest(3.0, ppmPeaksDiff)
GABAamp = resDiff[0,GABAPeakIdx]



# find correct GABA concentration using reference
GABAconc = GABAamp/CrAmp 
print 'GABA: '+ str(GABAconc)



def reconstruct(pfile, xmin=4.3, xmax=-0.8, sampling_rate=5000):
	"""
	Parameters
	----------

	pfile: name of pfile with data for reconstruction
	xmin: lower limit of x axis in ppm
	xmax: upper limit of x axis in ppm
	sampling_rate: 

	will add
	--------
	idx: slices to match xmin and xmax	
	data_dir: directory where data is
	"""

	data_dir = '/home/grace/spectro/data/'

	# read file, convert to npy format 
	data = io.get_data(os.path.join(data_dir, pfile))
	data = data.squeeze() # get rid of singleton dimensions

	np.save(os.path.join(data_dir,pfile[:-2]), data) # save in .npy format

	# process with-water and water-suppressed data
	print 'coil combine...'
	w_data, w_supp_data = coil_combine(data)
	w_data_off, w_supp_data_off, w_data_on, w_supp_data_on = two_echo_coil_combine(data)

	print w_data.shape
	#na = np.newaxis
	#w_data_off = w_data_off[na,:,:]
	print w_data_off.shape

	print 'apodize...'
	#ts_w = ut.apodize(nts.TimeSeries(np.mean(w_data, 1), sampling_rate=5000.))
	#ts_nonw = ut.apodize(nts.TimeSeries(np.mean(w_supp_data, 1), sampling_rate=5000.))
	ts_w_off = ut.apodize(nts.TimeSeries(np.mean(w_data_off, 1), sampling_rate=5000.))
	ts_nonw_off = ut.apodize(nts.TimeSeries(np.mean(w_supp_data_off, 1), sampling_rate=5000.))
	ts_w_on = ut.apodize(nts.TimeSeries(np.mean(w_data_on, 1), sampling_rate=5000.))
	ts_nonw_on = ut.apodize(nts.TimeSeries(np.mean(w_supp_data_on, 1), sampling_rate=5000.))

	print 'filter...'
	#f_ts_w = nta.FilterAnalyzer(ts_w, lb=1.8, ub=496, filt_order=128).fir
	#f_ts_nonw = nta.FilterAnalyzer(ts_nonw, lb=1.8, ub=496, filt_order=128).fir
	f_ts_w_off = nta.FilterAnalyzer(ts_w_off, lb=1.8, ub=496, filt_order=128).fir
	f_ts_nonw_off = nta.FilterAnalyzer(ts_nonw_off, lb=1.8, ub=496, filt_order=128).fir
	f_ts_w_on = nta.FilterAnalyzer(ts_w_on, lb=1.8, ub=496, filt_order=128).fir
	f_ts_nonw_on = nta.FilterAnalyzer(ts_nonw_on, lb=1.8, ub=496, filt_order=128).fir

	print 'spectral analyzer...'
	#S_w = nta.SpectralAnalyzer(f_ts_w, method=dict(NFFT=128), BW=6)
	#S_nonw = nta.SpectralAnalyzer(f_ts_nonw, method=dict(NFFT=128), BW=6)
	S_w_off = nta.SpectralAnalyzer(f_ts_w_off, method=dict(NFFT=128), BW=6)
	S_nonw_off = nta.SpectralAnalyzer(f_ts_nonw_off, method=dict(NFFT=128), BW=6)
	S_w_on = nta.SpectralAnalyzer(f_ts_w_on, method=dict(NFFT=128), BW=6)
	S_nonw_on = nta.SpectralAnalyzer(f_ts_nonw_on, method=dict(NFFT=128), BW=6)

	print 'multitaper...'
	#f_w, c_w = S_w.spectrum_multi_taper
	#f_nonw, c_nonw = S_nonw.spectrum_multi_taper
	f_w_off, c_w_off = S_w_off.spectrum_multi_taper
	f_nonw_off, c_nonw_off = S_nonw_off.spectrum_multi_taper
	f_w_on, c_w_on = S_w_on.spectrum_multi_taper
	f_nonw_on, c_nonw_on = S_nonw_on.spectrum_multi_taper

	# take only real values
	#w_sig = np.real(c_w)
	#nonw_sig = np.real(c_nonw)
	w_sig_off = np.real(c_w_off)
	nonw_sig_off = np.real(c_nonw_off)
	w_sig_on = np.real(c_w_on)
	nonw_sig_on = np.real(c_nonw_on)

	#corrected_off = normalize_water(w_sig_off, nonw_sig_off)
	#corrected_on = normalize_water(w_sig_on, nonw_sig_on)

	idx = slice(44,578)  # take indices 44:578.. matching to sage output was done by eye
	#scale_fac1 = np.mean(w_sig[0][idx]/nonw_sig[0][idx])
	#scale_fac2 = np.mean(w_sig[1][idx]/nonw_sig[1][idx])
	#scale_fac = scale_fac2
	#scale_fac = np.mean([scale_fac1, scale_fac2])

	scale_fac1_off = np.mean(w_sig_off[0][idx]/nonw_sig_off[0][idx])
	scale_fac2_off = np.mean(w_sig_off[1][idx]/nonw_sig_off[1][idx])
	scale_fac_off = scale_fac2_off
	scale_fac_off = np.mean([scale_fac1_off, scale_fac2_off])
	scale_fac1_on = np.mean(w_sig_on[0][idx]/nonw_sig_on[0][idx])
	scale_fac2_on = np.mean(w_sig_on[1][idx]/nonw_sig_on[1][idx])
	scale_fac_on = scale_fac2_on
	scale_fac_on = np.mean([scale_fac1_on, scale_fac2_on])

	#approx = w_sig/scale_fac
	#corrected = nonw_sig - approx # water correction
	##corrected /= np.mean(corrected)
	##approx.shape, corrected.shape
	#plot(approx[0][idx])
	#plot(nonw_sig[0][idx])


	approx_off = w_sig_off/scale_fac_off
	corrected_off = nonw_sig_off - approx_off # water correction
	approx_on = w_sig_on/scale_fac_on
	corrected_on = nonw_sig_on - approx_on # water correction
	#corrected /= np.mean(corrected)
	#approx.shape, corrected.shape

	#fig, ax = plt.subplots(1)
	#ax.plot((f_nonw[idx]-76)/128, cdiff[::2]/np.max(cdiff[::2]))
	# This correction is based on the assumption that for NAA, the resonance frequency is approximately 330 MHz and the 
	# value in ppm is 2.0. That means that to convert we should subtract 76 and divide by the normalizing factor (which is the
	# Larmor frequency of water
	#plot(((f_nonw[50:584]-76)/128)[::-1], (corrected[1][50:584]))
	#ax.plot((f_nonw_off[idx]-76)/128, np.real(corrected_off[1][idx])/np.max(np.real(corrected_off[1][idx])))
	#ax.plot((f_nonw_on[idx]-76)/128, np.real(corrected_on[1][idx])/np.max(np.real(corrected_on[1][idx])))

	amp_off = np.real(corrected_off[1][idx])/np.max(np.real(corrected_off[1][idx]))
	amp_on = np.real(corrected_on[1][idx])/np.max(np.real(corrected_on[1][idx]))
	#amp_off = np.real(corrected_off[1][idx])
	#amp_on = np.real(corrected_on[1][idx])


	npts = len(amp_on) 
	xAxis = np.arange(xmin,xmax,(xmax-xmin)/npts)

	#ax.plot(xAxis, amp_off)
	#ax.plot(xAxis, amp_on)

	figure(1)
	plot(xAxis, amp_off)
	plot(xAxis, amp_on)
	xlim(xmin, xmax)
	#ax.plot((f_nonw[idx]-76)/128, np.real(corrected[1][idx])/np.max(np.real(corrected[1][idx])))
	#ax[1].plot((f_nonw[idx]-76)/128, cdiff[::2]/np.max(cdiff[::2]) - np.real(corrected[1][idx])/np.max(np.real(corrected[1][idx])))
	#fig.set_size_inches([8,6])
	savefig(os.path.join(data_dir,pfile[:-2] + '.png'))

	#plt.show()
	return amp_on, amp_off

def quantify(pfile, xmin=4.3, xmax=-0.8):
	"""
	Paramaters:
	----------
	xmin: lower limit of x axis in ppm
        xmax: upper limit of x axis in ppm
        
	will add:
	idx
	"""
	data_dir = '/home/grace/spectro/data/'
	amp_on, amp_off = reconstruct(pfile)
	
	# calculate difference between on and off resonance data
	figure(1)

	npts = len(amp_on)
        xAxis = np.arange(xmin,xmax,(xmax-xmin)/npts)

	diff = array.array('f')
	for x in range(len(amp_on)):
		diff.append(amp_on[x]-amp_off[x])
	plot(xAxis,diff)
	xlim(xmin,xmax)
	savefig(os.path.join(data_dir,pfile[:-2] + '_diff.png'))

	#plt.show()

	# detect peaks in data - returns indices
	peaks1 = peakdetect(amp_off)
	peaks2 = peakdetect(amp_on)
	peaksDiff = peakdetect(diff)

	# convert to ppm
	ppmPeaks1 = array.array('f')
	ppmPeaks2 = array.array('f')
	ppmPeaksDiff = array.array('f')
	for i in range(len(peaks1)):
		ppmPeaks1.append(peaks1[i]*(xmax-xmin)/npts+xmin)
	for i in range(len(peaks2)):
		ppmPeaks2.append(peaks2[i]*(xmax-xmin)/npts+xmin)
	for i in range(len(peaksDiff)):
		ppmPeaksDiff.append(peaksDiff[i]*(xmax-xmin)/npts+xmin)


	GABApeakIdx = peak_nearest(3.0, ppmPeaksDiff) # peak from difference (echo1-echo2)
	CrpeakIdx = peak_nearest(3.0,ppmPeaks1) # GABA edited out of echo 1

	# linear model method - need to find locations of other major peaks
	#X_gaba = conv_lorentz(peaksDiff[GABApeakIdx], diff)
	#X_Cr = conv_lorentz(peaks1[CrpeakIdx], amp_off)


	# AUC method - see Sanacora paper
	GABA_AUC = scipy.integrate.simps(diff[int(((ppmPeaksDiff[GABApeakIdx]+0.15)-xmin)/(xmax-xmin)*npts): int(((ppmPeaksDiff[GABApeakIdx]-0.15)-xmin)/(xmax-xmin)*npts)])
	Cr_AUC = scipy.integrate.simps(amp_off[int(((ppmPeaks1[CrpeakIdx]+0.10)-xmin)/(xmax-xmin)*npts): int(((ppmPeaks1[CrpeakIdx]-0.10)-xmin)/(xmax-xmin)*npts)])

	# find correct GABA concentration using Creatine reference
	GABAconc = GABA_AUC/Cr_AUC
	print 'GABA: '+ str(GABAconc)

