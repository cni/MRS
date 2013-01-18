import os

import numpy as np
import matplotlib.pyplot as plt

import files as io
import nitime as nt
import nitime.timeseries as nts
import nitime.analysis as nta

import MRS.utils as ut

def coil_combine(data, on_off=0, w_idx=[1,2,3], sampling_rate=5000.0, NFFT=640):
    """
    Combine data across coils based on the amplitude of the water peak,
    according to:

    .. math::
        
        signal = sum_{i}(w(i)sig(i))

        w(i) = s(i)/(sqrt(sum(s(i)))
    
    where s(i) is the amplitude of the water peak in each coil 

    Parameters
    ----------
    data : float array
       The data as it comes from the scanner (read using the functions in
       files.py), with dimensions time x transients x off-/on-resonance x coils
    
    w_idx : tuple
       The indices to the non-water-suppressed transients. Per default we take
        the 2nd-4th transients. We dump the first one, because it seems to be
        quite different than the rest of them...
    
    sampling rate : float
        The sampling rate in Hz
    """
    # The transients are the second dimension in the data
    idxes_w = np.zeros(data.shape[1], dtype=bool)
    idxes_w[w_idx] = True
    # Data with water unsuppressed (first four transients - we throw away the
    # first one which is probably crap):
    w_data = data[:,np.where(idxes_w),:,:]
    # Data with water suppressed (the rest of the transients):
    idxes_nonw = np.zeros(data.shape[1], dtype=bool)
    idxes_nonw[np.where(~idxes_w)] = True
    idxes_nonw[0] = False
    w_supp_data = data[:,np.where(idxes_nonw),:,:]
    
    # Initialize time-series with this stuff
    ts_w = nt.TimeSeries(w_data.T, sampling_rate=sampling_rate)
    ts_w_supp = nt.TimeSeries(w_supp_data.T, sampling_rate=sampling_rate)
        
    # Push them through into spectral analysis. We'll use the Fourier
    # transformed version of this:
    S_w = nta.SpectralAnalyzer(ts_w)    
    f_w, psd_w = S_w.spectrum_fourier
    # Average across repeats:
    mean_psd = np.mean(psd_w.squeeze(), -1)
    mean_psd = np.mean(mean_psd, 2)
    # Average across off-/on-resonance (they're very similar anyway):
    mean_psd = mean_psd[..., on_off]
    w = np.power(mean_psd/np.sqrt(np.sum(mean_psd**2)),2)
    w = mean_psd/np.sqrt(np.sum(mean_psd))
    # Shorthand:
    na = np.newaxis
    # reshape to the number of dimensions in the data, so that you can broadcast
    w = w[:,na,na,na] #* 8 
    # Making sure the coil dimension is now first:
    weighted_w_data = np.sum(w * w_data.squeeze().T, 0)
    weighted_w_supp_data = np.sum(w * w_supp_data.squeeze().T, 0)
    return weighted_w_data, weighted_w_supp_data

def get_spectra(data, sampling_rate=5000.0,
                filt_method=dict(lb=1.8, ub=600., filt_order=128),
                spect_method=dict(NFFT=128, BW=6)):
    """
    Derive the spectra from MRS data

    Parameters
    ----------
    data : tuple
       (w_data, w_supp_data), where w_data contains data that is not
       water-suppressed (contains a huge water peak in the spectrum) and
       w_supp_data is water-suppressed. This is the output of `coil_combine`s 

    
    """
    w_data, w_supp_data = data

    ts_w = ut.apodize(nts.TimeSeries(np.mean(w_data, 1),
                                     sampling_rate=sampling_rate))
    
    ts_nonw = ut.apodize(nts.TimeSeries(np.mean(w_supp_data, 1),
                                        sampling_rate=sampling_rate))
    f_ts_w = nta.FilterAnalyzer(ts_w, **filt_method).fir
    f_ts_nonw = nta.FilterAnalyzer(ts_w, **filt_method).fir
    S_w = nta.SpectralAnalyzer(ts_w,
                               method=dict(NFFT=spect_method['NFFT']),
                               BW=spect_method['BW'])
    S_nonw = nta.SpectralAnalyzer(ts_nonw,
                                  method=dict(NFFT=spect_method['NFFT']),
                                  BW=spect_method['BW'])

    
    f_w, c_w = S_w.spectrum_multi_taper
    f_nonw, c_nonw = S_nonw.spectrum_multi_taper

    # Extract only the real part of the spectrum:
    w_sig = np.real(c_w)
    nonw_sig = np.real(c_nonw)

    # Return the tuple (f_w should be the same as f_nonw, so return only one of
    # them):
    return f_w, w_sig, nonw_sig


def normalize_water(w_sig, nonw_sig, idx=slice(44, 578)):
    """
    Normalize the water-suppressed signal by the signal that is not
    water-suppressed, to get rid of the residual water peak.
    
    """
    scale_fac = np.mean(w_sig[idx]/nonw_sig[idx])
    approx = w_sig/scale_fac
    corrected = nonw_sig - approx
    return corrected

if "__name__" == "__main__":
    # There will be some testing in here
    pass
