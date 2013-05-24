#!/usr/bin/env python

import argparse as arg

import numpy as np
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import nitime as nt

import MRS.files as fi
import MRS.analysis as ana
import MRS.utils as ut

parser = arg.ArgumentParser('Calculate MRS spectra from P files')

parser.add_argument('in_file', action='store', metavar='File', 
                    help='MRS P file')

parser.add_argument('--out_file', action='store', metavar='File', 
                    help='Output file (.csv)', default=False)
 
parser.add_argument('--sampling_rate', action='store', metavar='Float',
                    help='Sampling rate (default : 5000.0 Hz)',
                    default=5000.0)

parser.add_argument('--min_ppm', action='store', metavar='Float',
                    help='The minimal frequency (default: -0.7 ppm)',
                    default=-0.7)

parser.add_argument('--max_ppm', action='store', metavar='Float',
                    help='The minimal frequency (default: 4.3 ppm)',
                    default=4.3)

parser.add_argument('--broadening', action='store', metavar='Float',
                    help='Line broadening (Hz; default: 2)', default=2.0)

parser.add_argument('--zero_fill', action='store', metavar='Float',
                    help='Zero filling (number of points; default=100',
                    default=100)

parser.add_argument('--plot', action='store', metavar='Bool',
                    help='Whethere to produce a plot',
                    default=False)

in_args = parser.parse_args()

if __name__ == "__main__":
    # Get data from file: 
    data = fi.get_data(in_args.in_file)
    # Use the water unsuppressed data to combine over coils:
    w_data, w_supp_data = ana.coil_combine(data)
    # Remove residual effects of water
    w_supp_data = ana.subtract_water(w_data, w_supp_data)
    # Once we've done that, we only care about the water-suppressed data

    # Do spectral analysis:
    f_nonw, nonw_sig = ana.get_spectra(w_supp_data,
                                       line_broadening=float(in_args.broadening),
                                       zerofill=in_args.zero_fill,
                                       filt_method=None)

    f_ppm = ut.freq_to_ppm(f_nonw)
    idx0 = np.argmin(np.abs(f_ppm - in_args.min_ppm))
    idx1 = np.argmin(np.abs(f_ppm - in_args.max_ppm))
    idx = slice(idx1, idx0)
    # Convert from Hz to ppm and extract the part you are interested in.
    f_ppm = f_ppm[idx]

    print nonw_sig.shape
    
    # The first echo (off-resonance) is in the first output 
    echo1 = nonw_sig[:,0][:,idx]
    # The on-resonance is in the second:
    echo2 = nonw_sig[:,1][:,idx]

    # Pack it into a recarray:
    names = ('ppm', 'echo1', 'echo2', 'diff')
    formats = (float, float, float, float)
    dt = zip(names, formats)
    m_e1 = np.mean(echo1, 0)
    m_e2 = np.mean(echo2, 0)
    diff = m_e2 - m_e1

    if in_args.out_file:
        prep_arr = [(f_ppm[i], m_e1[i], m_e2[i], diff[i])
                              for i in range(len(f_ppm))]
        out_array = np.array(prep_arr, dtype=dt)

        # And save to output:
        mlab.rec2csv(out_array, in_args.out_file)

    if in_args.plot:
        fig, ax = plt.subplots(3)
        ax[0].plot(f_ppm, m_e1)
        ax[1].plot(f_ppm, m_e2)
        ax[2].plot(f_ppm, diff)
        for a in ax:
            a.invert_xaxis()
            a.set_xlabel('ppm')

        plt.show()
    
