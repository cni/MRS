#! /usr/bin/python

# attempting Wang 2006's method of calculating the GABA concentration 

import os
import array
import numpy
import scipy
import scipy.optimize
import scipy.integrate
import matplotlib.pyplot as plt
import sys
from frange import *

sage = os.environ['SAGE_DATABASE']
exam = str(sys.argv[1])

inc = -1* (5.1/534)
xrange = list(frange(4.3, -0.8, inc))
#print xrange
#print len(xrange)

newlines = array.array('c')
f1 = open(sage+'/'+exam+'/echo1/P11111.7_combine_pro.sdf', 'rb')
f2 = open(sage+'/'+exam+'/echo2/P11111.7_combine_pro.sdf', 'rb')

# if diff file was already created, open it
try:
   with open(sage+'/'+exam+'/echo1/diff.7_combine_pro.sdf', 'rb') as fdiff: pass
   fdiff = open(sage+'/'+exam+'/echo1/diff.7_combine_pro.sdf', 'rb')
   cdiff = array.array('f') # the 'c' is to identify diff from existing file
   cdiff.fromfile(fdiff, 1068)

except IOError as e:
   print 'Diff file not created yet... skipping.'


d1 = array.array('f')
d1.fromfile(f1, 1068)
ax1 = plt.subplot(3,1,1)
#print len(d1[::2])
#ax1.plot(d1[::2]) # plot real values only
ax1.plot(xrange,d1[::2]) # plot real values only
ax1.set_title('Echo 1')
ax1.set_xlim(4.3,-0.8)

d2 = array.array('f')
d2.fromfile(f2, 1068)
ax2 = plt.subplot(3,1,2)
ax2.plot(xrange,d2[::2])
ax2.set_title('Echo 2')
ax2.set_xlim(4.3,-0.8)

diff = d2
for i in range( len(diff) ):
    diff[i] -= d1[i]
ax3 = plt.subplot(3,1,3)
ax3.plot(xrange, diff[::2])
ax3.set_title('Diff')
ax3.set_ylim(-.02,.02)
ax3.set_xlim(4.3,-0.8)

plt.show()

# now calculate area under curve between 3.15ppm and 2.85ppm
# scale is 4.30 to -0.80, so 3.15 to 2.85 --> 120.4 tp 151.8
# EDIT: diff is really 1068 points, range is 240.8 to 303.6
auc = scipy.integrate.simps(diff[241:304])
print auc

#calculate the area of the Creatine peak at 3ppm in echo1 (check if this is "GABA inverted spectrum"?). Range = 3.1 to 2.9, so
aucCr = scipy.integrate.simps(d1[251:293])
print "AUC Cr: %f" %aucCr



# trying curve fitting (wang 2006)
plt.figure(2)
axdiff = plt.subplot(1,1,1)
axdiff.plot(xrange,cdiff[::2])
axdiff.set_xlim(4.3,-0.8)
axdiff.set_title('difference')

# curve fit
fitdiff = scipy.optimize.curve_fit(f, xrange, cdiff[::2])
axdiff.plot(xrange, fitdiff[::2])
plt.show()
