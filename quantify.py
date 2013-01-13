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
from peakdetect import *
from pylab import *
import array
sage = os.environ['SAGE_DATABASE']

def lorentzian(l,x):
	return l/(l**2+(x)**2)

# read file
exam = 'SM3714/SM_LStr2'
f1 = open(sage+'/'+exam+'/echo1/P11111.7_combine_pro.sdf', 'rb')
d1 = array.array('f')
d1.fromfile(f1, 1068)

d1real = d1[::2]

# plot fig
xAxis = np.arange(4.3,-0.8,-5.1/534)
#fig, ax = plt.subplots(1)
figure(1)
plot(range(534),d1real)
#plot(xAxis, d1real)
#ax.plot(xAxis,d1real) # plot real values only
#ax.set_title('Echo 1')
#ax.set_xlim(4.3,-0.8)
#plt.show()


# detect peaks
local_max = maximum_filter1d(d1real,100)
unique_max = list(set(local_max))
# find index of maxima
peak_x=array.array('f')
for idx in range(len(unique_max)):
	peak_x.append(d1real.index(unique_max[idx]))
# if indices are very close together, take the max of that cluster
maxmax = array.array('f')
for idx in range(len(peak_x)):
	cluster=array.array('f')
	[cluster.append(peak_x[i]) for i in range(len(peak_x)) if peak_x[idx]-10<peak_x[i]<peak_x[idx]+10]
	print cluster
	max_x=array.array('f')
	for idx in range(len(cluster)):
		max_x.append(d1real[int(cluster[idx])])
		print d1real[int(cluster[idx])]
	maxmax.append(max(max_x))
uniquemaxmax = list(set(maxmax))	
unique_peak=array.array('f')
for idx in range(len(uniquemaxmax)):
        unique_peak.append(d1real.index(uniquemaxmax[idx]))
print unique_peak

for idx in range(len(unique_peak)):
	plot(unique_peak[idx],d1real[int(unique_peak[idx])], 'ro')
#plt.show()


# model peaks convolve delta and lorentzian function 
lorentz=array.array('f')
for x in np.arange(-1,1,0.01):
	lorentz.append(lorentzian(0.1,x))
#figure(2)
#plot(np.arange(-1,1,0.01),lorentz)
#plt.show()

# for each unique peak found before, construct delta function
deltas=np.zeros((len(unique_peak),534))
for i in range(len(unique_peak)):
	deltas[i][int(unique_peak[i])]=1

delta_mat = np.asmatrix(deltas)
print delta_mat
# convolve with lorentzian
deltas_con = np.zeros((1,534))
for i in range(len(unique_peak)):
	print len(deltas[i])
	deltas_c = deltas[i]*lorentz
	plot(deltas_c)
	print len(deltas_c)
	print len(deltas_con)
	deltas_con=np.vstack([deltas_con, deltas_c])

#initial = 0.1, 0.1 # initial guess
#params, _ = opt.leastsq(err_func, initial, args=(xAxis,d1,lorentzian))


#xopt = fmin(lorentzian, x0, maxiter=300)
#print xopt



