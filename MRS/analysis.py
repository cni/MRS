import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import nitime as nt
import nitime.timeseries as nts
import nitime.analysis as nta
import scipy.fftpack as fft

import MRS.utils as ut


def separate_signals(data, w_idx=[1,2,3]):
   """
   Separate the water and non-water data from each other

   Parameters
   ----------
   data : nd array
      FID signal with shape (transients, echos, coils, time-points)

   w_idx : list (optional)
      Indices into the 'transients' (0th) dimension of the data for the signal
      that is not water-suppressed

   Returns
   -------
   water_data, w_supp_data : tuple
       The first element is an array with the transients in the data in which
       no water suppression was applied. The second element is an array with
       the transients in which water suppression was applied
   """

   # The transients are the first dimension in the data
   idxes_w = np.zeros(data.shape[0], dtype=bool)
   idxes_w[w_idx] = True
   # Data with water unsuppressed (first four transients - we throw away the
   # first one which is probably crap):
   w_data = data[np.where(idxes_w)]
   # Data with water suppressed (the rest of the transients):
   idxes_nonw = np.zeros(data.shape[0], dtype=bool)
   idxes_nonw[np.where(~idxes_w)] = True
   idxes_nonw[0] = False
   w_supp_data = data[np.where(idxes_nonw)]

   return w_data, w_supp_data


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
    coils is combined as:

    .. math :: 

        S = \sum_{i=1}^{n}{w_i * S_i}

    In addition, a phase correction is applied, so that all coils have the same
    phase.
  
    References
    ----------
    .. [Wald1997] Wald, L. and Wright, S. (1997). Theory and application of
       array coils in MR spectroscopy. NMR in Biomedicine, 10: 394-410.

    .. [Keeler2005] Keeler, J (2005). Understanding NMR spectroscopy, second
       edition. Wiley (West Sussex, UK).
    
    """
    w_data, w_supp_data = separate_signals(data, w_idx)     
    fft_w = fft.fft(w_data)
    # We use the water peak (the 0th frequency) as the reference (averaging
    # across transients and echos):
    zero_freq_w = np.mean(np.mean(np.abs(fft_w[..., 0]), 0), 0)
   
    # This is the weighting by SNR (equation 29 in the Wald paper):
    zero_freq_w_across_coils = np.sqrt(np.sum(zero_freq_w**2, -1))
    w = zero_freq_w/zero_freq_w_across_coils[...,np.newaxis]

    # Next, we make sure that all the coils have the same phase. We will use
    # the phase of this peak to align the phases: 
    zero_phi_w = np.angle(w_data[..., 0])
    zero_phi_w = np.mean(np.mean(zero_phi_w, 0), 0)
    # This recalculates the weight with the phase alignment (see page 397 in
    # Wald paper):
    w = w * np.exp(-1j * zero_phi_w) 
    # Multiply each one of them by it's weight and average across coils (2nd
    # dim). This makes sure that you are roughly 0 phased for the water peak
    na = np.newaxis  # Short-hand
    weighted_w_data = np.mean(w[na, na, :, na] * w_data, axis=2)
    weighted_w_supp_data = np.mean(w[na, na, :, na] * w_supp_data, axis=2)
    
    def normalize_this(x):
       return  x * (x.shape[-1] / np.sum(x))

    weighted_w_data = normalize_this(weighted_w_data)
    weighted_w_supp_data = normalize_this(weighted_w_supp_data)
    return weighted_w_data, weighted_w_supp_data 


def get_spectra(data, filt_method = dict(lb=0.1, filt_order=256),
                spect_method=dict(NFFT=1024, n_overlap=1023, BW=2),
                phase_zero=None, line_broadening=None, zerofill=None):
    """
    Derive the spectra from MRS data

    Parameters
    ----------
    data : nitime TimeSeries class instance or array
        Time-series object with data of shape (echos, transients, time-points),
        containing the FID data. If an array is provided, we will assume that a
        sampling rate of 5000.0 Hz was used

    filt_method : dict
        Details for the filtering method. A FIR zero phase-delay method is used
        with parameters set according to these parameters
        
    spect_method : dict
        Details for the spectral analysis. Per default, we use 

    line_broadening : float
        Linewidth for apodization (in Hz).

    zerofill : int
        Number of bins to zero fill with.
        
    Returns
    -------
    f, spectrum_water, spectrum_water_suppressed :

    f is the center frequency of the frequencies represented in the
    spectra. The first spectrum is for the data with water not suppressed and
    the s

    Notes
    -----
    This function performs the following operations:

    1. Filtering.
    2. Apodizing/windowing. Optionally, this is done with line-broadening (see
    page 92 of Keeler2005_.
    3. Spectral analysis.
    
    Notes
    -----
    
    .. [Keeler2005] Keeler, J (2005). Understanding NMR spectroscopy, second
       edition. Wiley (West Sussex, UK).

    """
    if not isinstance(data, nt.TimeSeries):
       data = nt.TimeSeries(data, sampling_rate=5000.0)  
    if filt_method is not None:
        filtered = nta.FilterAnalyzer(data, **filt_method).fir
    else:
        filtered = data
    if line_broadening is not None: 
       lbr_time = line_broadening * np.pi  # Conversion from Hz to
                                           # time-constant, see Keeler page 94 
    else:
       lbr_time = 0

    apodized = ut.line_broadening(filtered, lbr_time)
   
    if zerofill is not None:
         new_apodized = np.concatenate([apodized.data,
                    np.zeros(apodized.shape[:-1] + (zerofill,))], -1)

         apodized = nt.TimeSeries(new_apodized,
                                  sampling_rate=apodized.sampling_rate)

    S = nta.SpectralAnalyzer(apodized,
                             method=dict(NFFT=spect_method['NFFT'],
                                         n_overlap=spect_method['n_overlap']),
                             BW=spect_method['BW'])
    
    f, c = S.spectrum_fourier

    return f, c

def subtract_water(w_sig, nonw_sig, idx):
    """
    Subtract the residual water signal from the 
    Normalize the water-suppressed signal by the signal that is not
    water-suppressed, to get rid of the residual water peak.

    """
    mean_water = np.mean(w_sig, 0)

    non_w_sig

    water_only = mean_water - mean_supp
    scale_factor = np.sum(water_only)/np.sum(mean_supp)

    scale_fac = np.mean(w_sig[idx]/nonw_sig[idx])
    approx = w_sig/scale_fac
    corrected = nonw_sig - approx
    return corrected


def fit_lorentzian():
   pass
