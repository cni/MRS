import numpy as np
import scipy.stats as stats
import nibabel as nib
import warnings

import MRS.analysis as ana
import MRS.utils as ut
import MRS.freesurfer as fs

from scipy import interpolate

def naa_correct(G):

    """
    This function resets the fits and corrects shifts in the spectra.
    It uses uses the NAA peak at 2.0ppm as a guide to replaces the existing
    f_ppm values! 
    """
    G.reset_fits()

    # calculate diff
    diff = np.mean(G.diff_spectra, 0)
    # find index of NAA peak in diff spectrum, that is between 3 and 1ppm
    temp_diff = np.mean(G.diff_spectra, 0)
    temp_diff[slice(0,np.min(np.where(G.f_ppm<3)))]=0
    temp_diff[np.max(np.where(G.f_ppm>1)):]=0 
    idx = np.argmin(temp_diff)
    adjust_by=(float(idx)/len(diff))*(np.max(G.f_ppm)-
                                      np.min(G.f_ppm))
    NAA_ppm = np.max(G.f_ppm)-adjust_by 
    
    # determine how far spectrum is shifted
    NAA_shift = 2.0-NAA_ppm
    
    # correct
    G.f_ppm = G.f_ppm + NAA_shift

    # tag as corrected
    G.naa_corrected = True
    
def baseline_correct(G):

    """
    This function zeroes the baseline from 2.5ppm upwards 
   
    """
    # define ppm ranges that are known to be at baseline, get indices
    baseidx =[]
    baseidx.extend(range(np.min(np.where(G.f_ppm<5.0)),np.max(np.where(G.f_ppm>4.0))+1))
    baseidx.extend(range(np.min(np.where(G.f_ppm<3.5)),np.max(np.where(G.f_ppm>3.2))+1))
    baseidx.extend(range(np.min(np.where(G.f_ppm<2.8)),np.max(np.where(G.f_ppm>2.5))+1))

    G.diff = np.mean(G.diff_spectra,0)
    # find x and y values at those indices
    yArr=np.real(G.diff[baseidx])
    baseppm = G.f_ppm[baseidx]
    # filter out anything above the new max
    adjbaseppm =[baseppm[i] for i in np.where(baseppm<=np.max(G.f_ppm))[0]]
    
    # spline
    f = interpolate.interp1d(adjbaseppm[::-1], yArr[::-1], kind='linear', bounds_error=True, fill_value=0)
    
    fitidxmax = np.where(G.f_ppm<np.max(adjbaseppm))[0]
    fitidxmin = np.where(G.f_ppm>np.min(adjbaseppm))[0]
    fitidx = list(set(fitidxmax) & set(fitidxmin))
    
    basefit = f(G.f_ppm[fitidx])#[::-1]
    adjusted = G.diff[fitidx]-basefit#[::-1]

    G.diff_corrected = G.diff
    G.diff_corrected[fitidx] = adjusted
    
    # tag as corrected
    G.baseline_corrected = True

