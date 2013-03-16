import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import files as io
import nitime as nt
import nitime.timeseries as nts
import nitime.analysis as nta
import scipy.fftpack as fft

import MRS.utils as ut

def coil_combine(data, w_idx=[1,2,3]):
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
    
    fft_w = fft.fft(w_data).squeeze()
    fft_w_supp = fft.fft(w_supp_data).squeeze()

    # We use the water peak (the 0th frequency) as the reference:
    zero_freq_w = np.abs(fft_w[0])

    # This is the weighting by SNR (equation 29 in the Wald paper):
    zero_freq_w_across_coils = np.sqrt(np.sum(zero_freq_w**2,-1))
    w = zero_freq_w/zero_freq_w_across_coils[...,np.newaxis]

    # We average across echos and repeats:
    w = np.mean(np.mean(w,0),0)

    # We will use the phase of this peak to align the phases:
    zero_phi_w = np.angle(fft_w[0])
    zero_phi_w = np.mean(np.mean(zero_phi_w,0), 0)

    # This recalculates the weight with the phase alignment (see page 397 in
    # Wald paper):
    w = w * np.exp(-1j * zero_phi_w) 

    # Dot product each one of them and ifft back into the time-domain
    na = np.newaxis # Short-hand
    weighted_w_data = fft.ifft(np.sum(w[na, na, na, :] * fft_w,
                                      axis=-1), axis=-1)
    weighted_w_supp_data = fft.ifft(np.sum(w[na, na, na, :] * fft_w_supp,
                                           axis=-1),axis=-1)
    # Transpose, so that the time dimension is last:
    w_out = np.squeeze(weighted_w_data).T
    w_supp_out = np.squeeze(weighted_w_supp_data).T
    # Normalize to sum to number of measurements:
    w_out = w_out * (w_out.shape[-1] / np.sum(w_out))
    w_supp_out = w_supp_out * (w_supp_out.shape[-1] / np.sum(w_supp_out))
    return w_out, w_supp_out 

def get_spectra(data, sampling_rate=5000.0,
                spect_method=dict(NFFT=1024, n_overlap=512,
                                  #detrend=mlab.detrend_linear,
                                  BW=12),
                filt_method = dict(lb=0.1, filt_order=256)):
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

    ts_w = ut.apodize(nta.FilterAnalyzer(
        nts.TimeSeries(w_data, sampling_rate=sampling_rate),
        **filt_method).fir)
    
    ts_nonw = ut.apodize(nta.FilterAnalyzer(
        nts.TimeSeries(w_supp_data,sampling_rate=sampling_rate),
        **filt_method).fir)

    S_w = nta.SpectralAnalyzer(ts_w,
                               method=dict(NFFT=spect_method['NFFT'],
                                           n_overlap=spect_method['n_overlap']),
                               BW=spect_method['BW'])

    S_nonw = nta.SpectralAnalyzer(ts_nonw,
                                  method=dict(NFFT=spect_method['NFFT'],
                                           n_overlap=spect_method['n_overlap']),
                                  BW=spect_method['BW'])

    f_w, c_w = S_w.spectrum_fourier
    f_nonw, c_nonw = S_nonw.spectrum_fourier

    # Return the tuple (f_w should be the same as f_nonw, so return only one of
    # them):
    return f_w, np.real(c_w), np.real(c_nonw)


def normalize_water(w_sig, nonw_sig, idx):
    """
    Normalize the water-suppressed signal by the signal that is not
    water-suppressed, to get rid of the residual water peak.

    Might not be necessary if appropriate filtering is applied to the signal.
    
    """
    scale_fac = np.mean(w_sig[idx]/nonw_sig[idx])
    approx = w_sig/scale_fac
    corrected = nonw_sig - approx
    return corrected

if "__name__" == "__main__":
    # There will be some testing in here
    pass
