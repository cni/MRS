#!/usr/bin/env python

import argparse as arg

import numpy as np
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats

import MRS.analysis as ana
import MRS.utils as ut
import MRS.api as mrs

parser = arg.ArgumentParser('Calculate MRS spectra from Nifti files')

parser.add_argument('in_file', action='store', metavar='File', 
                    help='MRS Nifti file (see http://cni.github.io/MRS/doc/_build/html/data.html for specification of the data format)')

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
                    help='Line broadening (Hz; default: 5)', default=5.0)

parser.add_argument('--zero_fill', action='store', metavar='Float',
                    help='Zero filling (number of points; default=100',
                    default=100)

parser.add_argument('--plot', action='store', metavar='Bool',
                    help='Whether to produce a plot',
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
    names = ('ppm', 'echo_on', 'echo_off', 'diff')
    formats = (float, float, float, float)
    dt = zip(names, formats)
    m_e1 = np.mean(G.echo_on, 0)
    m_e2 = np.mean(G.echo_off, 0)
    diff = np.mean(G.diff_spectra, 0)
    
    if in_args.out_file:
        prep_arr = [(G.f_ppm[i], m_e1[i], m_e2[i], diff[i])
                              for i in range(len(G.f_ppm))]
        out_array = np.array(prep_arr, dtype=dt)

        # And save to output:
        mlab.rec2csv(out_array, in_args.out_file)

    
    if in_args.plot:
        G.fit_gaba()
        fig, ax = plt.subplots(2)
        ax[0].plot(G.f_ppm[G.idx], np.mean(G.sum_spectra,0)[G.idx])
        ax[0].plot(G.f_ppm[G.cr_idx], stats.nanmean(G.creatine_model, 0), 'r',
                   label='Creatine model')
        ax[0].text(.8, .8, 'Creatine AUC: %.2f'%stats.nanmean(G.creatine_auc),
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax[0].transAxes)
        ax[0].set_title('Sum spectra (echo on + echo off)')

        ax[1].plot(G.f_ppm[G.idx], np.mean(G.diff_spectra, 0)[G.idx])
        ax[1].plot(G.f_ppm[G.gaba_idx], stats.nanmean(G.gaba_model, 0), 'r',
                   label='Gaba model')
        ax[1].text(.8, .8, 'GABA AUC: %.2f'%stats.nanmean(G.gaba_auc),
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax[1].transAxes)
        
        ax[1].set_title('Difference spectra (echo on - echo off)')
        for a in ax:
            a.invert_xaxis()
            a.set_xlabel('ppm')

        plt.show()

    
