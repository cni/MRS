#!/usr/bin/env python

import argparse as arg

import numpy as np
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import nitime as nt

import MRS.analysis as ana
import MRS.utils as ut
import MRS.api as mrs

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
                    help='Line broadening (Hz; default: 2)', default=5.0)

parser.add_argument('--zero_fill', action='store', metavar='Float',
                    help='Zero filling (number of points; default=100',
                    default=100)

parser.add_argument('--plot', action='store', metavar='Bool',
                    help='Whethere to produce a plot',
                    default=False)

in_args = parser.parse_args()

if __name__ == "__main__":
    G = mrs.GABA(in_args.in_file,
                 line_broadening=float(in_args.broadening),
                 zerofill=float(in_args.zero_fill),
                 filt_method=None,
                 min_ppm=in_args.min_ppm,
                 max_ppm=in_args.max_ppm)
    
    # Pack it into a recarray:
    names = ('ppm', 'echo1', 'echo2', 'diff')
    formats = (float, float, float, float)
    dt = zip(names, formats)
    m_e1 = np.mean(G.echo1, 0)
    m_e2 = np.mean(G.echo2, 0)
    diff = m_e2 - m_e1

    if in_args.out_file:
        prep_arr = [(G.f_ppm[i], m_e1[i], m_e2[i], diff[i])
                              for i in range(len(G.f_ppm))]
        out_array = np.array(prep_arr, dtype=dt)

        # And save to output:
        mlab.rec2csv(out_array, in_args.out_file)

    G.fit_gaba()
    
    if in_args.plot:
        fig, ax = plt.subplots(3)
        ax[0].plot(G.f_ppm, m_e1)
        ax[0].plot(G.f_ppm[G.cr_idx], np.mean(G.creatine_model, 0), 'r')
        ax[1].plot(G.f_ppm, m_e2)
        ax[2].plot(G.f_ppm, diff)
        ax[2].plot(G.f_ppm[G.gaba_idx], np.mean(G.gaba_model, 0), 'r')
        for a in ax:
            a.invert_xaxis()
            a.set_xlabel('ppm')

        plt.show()
    
