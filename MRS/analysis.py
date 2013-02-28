import os

import numpy as np
import matplotlib.pyplot as plt

import files as io
import nitime as nt
import nitime.timeseries as nts
import nitime.analysis as nta
import scipy.fftpack as fft

import MRS.utils as ut

def coil_combine(data, w_idx=[0,1,2,3]):
    """
    Combine data across coils based on the amplitude of the water peak,
    according to:

    .. math::
        
        signal = sum_{i}(w(i)sig(i))

        w(i) = s(i)/(sqrt(sum(s(i)))
        
    where s(i) is the amplitude of the water peak in each coil.

    In addition, we apply a phase-correction, so that all the phases of the
    signals from each coil are 0

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

    Notes
    -----
    Following [Wald1997]_, we compute weights on the different coils based on
    the amplitudes and phases of the water peak. The signal from different
    coils is ultimately combined as

    .. math :: 

        S = \sum_{i=1}^{n}{w_i * S_i}

    Where w_i are weights on the individual coils. To derive $w_i$, we use
    eqaution 29/30 from the Wald paper:

    .. math ::

         S = 
   
    References
    ----------
    .. [Wald1997] Wald, L. and Wright, S. (1997). Theory and application of
       array coils in MR spectroscopy. NMR in Biomedicine, 10: 394-410.
    
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

    fft_w = fft.fft(w_data)
    fft_w_supp = fft.fft(w_supp_data)

    # We use the water peak (the 0th frequency) as the reference:
    zero_freq_w = np.abs(fft_w[0])

    # This is the weighting by SNR (equation 29 in the Wald paper):
    zero_freq_w_across_coils = np.sqrt(np.sum(zero_freq_w**2,-1))
    w = zero_freq_w/zero_freq_w_across_coils[...,np.newaxis]

    # We average across 
    w = np.mean(np.mean(w,0),0)

    # We will use the phase of this peak to align the phases:
    zero_phi_w = np.angle(fft_w[0])
    zero_phi_w = np.mean(np.mean(zero_phi_w,0), 0)

    # This recalculates the weight with the phase alignment (see page 397 in
    # Wald paper):
    w = w * np.exp(-1j * zero_phi_w) 

    # Dot product each one of them and ifft back into the time-domain
    na = np.newaxis # Short-hand
    weighted_w_data = fft.ifft(np.sum(w[na, na, na, :] * fft_w, -1))
    weighted_w_supp_data = fft.ifft(np.sum(w[na, na, na, :] * fft_w_supp, -1))
    # Transpose, so that the time dimension is last:
    return np.squeeze(weighted_w_data).T, np.squeeze(weighted_w_supp_data).T


def get_spectra(data, sampling_rate=5000.0,
                spect_method=dict(NFFT=128, BW=2)):
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

    S_w = nta.SpectralAnalyzer(ts_w,
                               method=dict(NFFT=spect_method['NFFT']),
                               BW=spect_method['BW'])

    S_nonw = nta.SpectralAnalyzer(ts_nonw,
                                  method=dict(NFFT=spect_method['NFFT']),
                                  BW=spect_method['BW'])

    
    f_w, c_w = S_w.spectrum_multi_taper
    f_nonw, c_nonw = S_nonw.spectrum_multi_taper

    # Return the tuple (f_w should be the same as f_nonw, so return only one of
    # them):
    return f_w, c_w, c_nonw


def normalize_water(w_sig, nonw_sig, idx):
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
