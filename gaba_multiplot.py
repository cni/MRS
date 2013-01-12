#! /usr/bin/python

# plots multiple spectra on the same plot
# takes folders with echo1 and echo2 subfolders as arguments
# e.g. usage: python gaba_multiplot.py folder1 folder2 folder3 

import os
import array
import numpy
import matplotlib.pyplot as plt
import sys

sage = os.environ['SAGE_DATABASE']
for exam in range(1,len(sys.argv)):
	print 'plotting %s' % sys.argv[exam]

	newlines = array.array('c')
	f1 = open(sage+'/'+sys.argv[exam]+'/echo1/P11111.7_combine_pro.sdf', 'rb')
	f2 = open(sage+'/'+sys.argv[exam]+'/echo2/P11111.7_combine_pro.sdf', 'rb')

	d1 = array.array('f')
	d1.fromfile(f1, 1068)

	d2 = array.array('f')
	d2.fromfile(f2, 1068)

	diff1 = d2
	for i in range( len(diff1) ):
	    diff1[i] -= d1[i]

	ax = plt.subplot(1,1,1)
	ax.plot(diff1[::2])

ax.set_title('Diffs')
ax.set_ylim(-.1,.1)
ax.set_xlim(0,534)

plt.show()
