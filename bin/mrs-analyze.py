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

parser.add_argument('out_file', action='store', metavar='File', 
                    help='Output file (.csv)')
 
parser.add_argument('--sampling_rate', action='store', metavar='Float',
                    help='Sampling rate (default : 5000.0 Hz)',
                    default=5000.0)

parser.add_argument('--min_ppm', action='store', metavar='Float',
                    help='The minimal frequency (default: -0.7 ppm)',
                    default=-0.7)

parser.add_argument('--max_ppm', action='store', metavar='Float',
                    help='The minimal frequency (default: 4.3 ppm)',
                    default=4.3)

parser.add_argument('--plot', action='store', metavar='Bool',
                    help='Whethere to produce a plot',
                    default=False)

in_args = parser.parse_args()

if __name__ == "__main__":
    # Get data from file: 
    data = fi.get_data(in_args.in_file)
    # Use the water unsuppressed data to combine over coils:
    w_data, w_supp_data = ana.coil_combine(data.squeeze())
    # Once we've done that, we only care about the water-suppressed data
    f_nonw, nonw_sig = ana.get_spectra(nt.TimeSeries(w_supp_data,
                                        sampling_rate=in_args.sampling_rate))

    f_ppm = ut.freq_to_ppm(f_nonw)
    idx0 = np.argmin(np.abs(f_ppm - in_args.min_ppm))
    idx1 = np.argmin(np.abs(f_ppm - in_args.max_ppm))
    idx = slice(idx1, idx0)
    # Convert from Hz to ppm and extract the part you are interested in.
    f_ppm = f_ppm[idx]

    # The first echo (off-resonance) is in the first output 
    echo1 = nonw_sig[0][:,idx]
    # The on-resonance is in the second:
    echo2 = nonw_sig[1][:,idx]
    
    # Pack it into a recarray:
    names = ('ppm', 'echo1', 'echo2', 'diff')
    formats = (float, float, float, float)
    dt = zip(names, formats)
    m_e1 = np.mean(echo1, 0)
    m_e2 = np.mean(echo2, 0)
    diff = m_e2 - m_e1
    prep_arr = [(f_ppm[i], m_e1[i], m_e2[i], diff[i]) for i in range(len(f_ppm))]
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
    
