#! /usr/bin/python

# Soher 1996, "quantitation of automated single voxel proton MRS using cerebral water as an internal reference"

import os
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
from MRS.utils import *
from MRS.peaks import * 

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



