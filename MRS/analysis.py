import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import nitime as nt
import nitime.timeseries as nts
import nitime.analysis as nta
import scipy.fftpack as fft
import scipy.integrate as spi
from scipy.integrate import trapz, simps

import MRS.leastsqbound as lsq
import MRS.utils as ut
import MRS.optimize as mopt

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


def coil_combine(data, w_idx=[1,2,3], coil_dim=2, sampling_rate=5000.):
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
       files.py), with shape (transients, echos, coils, time points)
    
    w_idx : tuple
       The indices to the non-water-suppressed transients. Per default we take
        the 2nd-4th transients. We dump the first one, because it seems to be
        quite different than the rest of them...

    coil_dim : int
        The dimension on which the coils are represented. Default: 2

    sampling rate : float
        The sampling rate in Hz. Default : 5000.

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

    fft_w = np.fft.fftshift(fft.fft(w_data))
    fft_w_supp = np.fft.fftshift(fft.fft(w_supp_data))
    freqs_w = np.linspace(-sampling_rate/2.0,
                          sampling_rate/2.0,
                          w_data.shape[-1])
    
    # To determine phase and amplitude, fit a Lorentzian line-shape to each
    # coils data in each trial: 
    # No bounds except for on the phase:
    bounds = [(None,None),
              (0,None),
              (0,None),
              (-np.pi, np.pi),
              (None,None),
              (None, None)]

    n_params = len(bounds)
    params = np.zeros(fft_w.shape[:-1] + (n_params,))

    # Let's fit a Lorentzian line-shape to each one of these:
    for repeat in range(w_data.shape[0]):
       for echo in range(w_data.shape[1]):
          for coil in range(w_data.shape[2]):
             sig = fft_w[repeat, echo, coil]
             # Use the private function to do this:
             params[repeat, echo, coil] = _do_lorentzian_fit(freqs_w,
                                                             sig, bounds)


    # The area parameter stands for the magnitude:
    area_w = params[..., 1]
    # This is the weighting by SNR (equation 29 in the Wald paper):
    norm_factor = np.sqrt(np.sum(area_w**2, -1))
    amp_weight = area_w/norm_factor[...,np.newaxis]

    # Next, we make sure that all the coils have the same phase. We will use
    # the phase of the Lorentzian to align the phases: 
    zero_phi_w = params[..., 3]

    # This recalculates the weight with the phase alignment (see page 397 in
    # Wald paper):
    weight = amp_weight * np.exp(-1j * zero_phi_w) 

    # Average across repetitions and echos:
    final_weight = np.mean(weight, axis=(0,1))

    # Multiply each one of the signals by its coil-weights and average across
    # coils:
    na = np.newaxis  # Short-hand

    # Collapse across coils for the combination in both the water 
    weighted_w_data = np.mean(np.fft.ifft(np.fft.fftshift(
       final_weight[na, na, :, na] * fft_w)), coil_dim)
    weighted_w_supp_data = np.mean(np.fft.ifft(np.fft.fftshift(
       final_weight[na, na, : ,na] * fft_w_supp)) , coil_dim)

    # Normalize each series by the sqrt(rms):
    def normalize_this(x):
       return  x * (x.shape[-1] / (np.sum(np.abs(x))))

    weighted_w_data = normalize_this(weighted_w_data)
    weighted_w_supp_data = normalize_this(weighted_w_supp_data)
    # Squeeze in case that some extraneous dimensions were introduced (can
    # happen for SV data, for example)
    return weighted_w_data.squeeze(), weighted_w_supp_data.squeeze()


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

def subtract_water(w_sig, w_supp_sig):
    """
    Subtract the residual water signal from the 
    Normalize the water-suppressed signal by the signal that is not
    water-suppressed, to get rid of the residual water peak.

    Parameters
    ----------
    w_sig : array with shape (n_reps, n_echos, n_points)
       A signal with water unsupressed

    w_supp_sig :array with shape (n_reps, n_echos, n_points)
       A signal with water suppressed.

    Returns
    -------
    The water suppressed signal with the additional subtraction of a scaled
    version of the signal that is presumably just due to water.

    """
    mean_nw = np.mean(w_supp_sig,0)
    water_only = np.mean(w_sig - mean_nw, 0)
    mean_water = np.mean(w_sig, 0)

    scale_factor = water_only/mean_nw

    corrected = w_supp_sig - water_only/scale_factor[...,0,np.newaxis]
    return corrected


def fit_lorentzian(spectra, f_ppm, lb=2.6, ub=3.6):
   """
   Fit a lorentzian function to the sum spectra to be used for estimation of
   the creatine peak, for phase correction of the difference spectra and for
   outlier rejection.
   
   Parameters
   ----------
   spectra : array of shape (n_transients, n_points)
      Typically the sum of the on/off spectra in each transient.

   f_ppm : array

   lb, ub: floats
      In ppm, the range over which optimization is bounded
   
   """
   # We are only going to look at the interval between lb and ub
   idx = ut.make_idx(f_ppm, lb, ub)
   n_points = np.abs(idx.stop - idx.start) 
   n_params = 6
   # Set the bounds for the optimization
   bounds = [(lb,ub),
             (0,None),
             (0,None),
             (-np.pi, np.pi),
             (None,None),
             (None, None)]

   model = np.empty((spectra.shape[0], n_points))
   signal = np.empty((spectra.shape[0], n_points))
   params = np.empty((spectra.shape[0], n_params))
   for ii, xx in enumerate(spectra):
      # We fit to the real spectrum:
      signal[ii] = np.real(xx[idx])
      params[ii] = _do_lorentzian_fit(f_ppm[idx], np.real(signal[ii]),
                                      bounds=bounds)
      
      model[ii] = ut.lorentzian(f_ppm[idx], *params[ii])
   
   return model, signal, params


def _do_lorentzian_fit(freqs, signal, bounds=None):
   """

   Helper function, so that Lorentzian fit can be generalized to different
   frequency scales (Hz and ppm).
   
   """
   # Use the signal for a rough estimate of the parameters for initialization:
   max_idx = np.argmax(np.real(signal))
   max_sig = np.max(np.real(signal))
   initial_f0 = freqs[max_idx]
   half_max_idx = np.argmin(np.abs(np.real(signal) - max_sig/2))
   initial_hwhm = np.abs(initial_f0 - freqs[half_max_idx])
   # Everything should be treated as real, except for the phase!
   initial_ph = np.angle(signal[signal.shape[-1]/2.])

   initial_off = np.min(np.real(signal))
   initial_drift = 0
   initial_a = (np.sum(np.real(signal)[max_idx:max_idx +
                              np.abs(half_max_idx)*2]) ) * 2

   initial = (initial_f0,
              initial_a,
              initial_hwhm,
              initial_ph,
              initial_off,
              initial_drift)

   params, _ = lsq.leastsqbound(mopt.err_func, initial,
                                args=(freqs, np.real(signal), ut.lorentzian),
                                bounds=bounds)
   return params


def fit_two_lorentzian(spectra, f_ppm, lb=2.6, ub=3.6):
   """
   Fit a lorentzian function to the sum spectra to be used for estimation of
   the creatine peak, for phase correction of the difference spectra and for
   outlier rejection.
   
   Parameters
   ----------
   spectra : array of shape (n_transients, n_points)
      Typically the sum of the on/off spectra in each transient.

   f_ppm : array

   lb, ub: floats
      In ppm, the range over which optimization is bounded
   
   """
   # We are only going to look at the interval between lb and ub
   idx = ut.make_idx(f_ppm, lb, ub)
   n_points = np.abs(idx.stop - idx.start) 
   n_params = 10 # Lotsa params!
   # Set the bounds for the optimization
   bounds = [(lb,ub),
             (lb,ub),
             (0,None),
             (0,None),
             (0,None),
             (0,None),
             (-np.pi, np.pi),
             (-np.pi, np.pi),
             (None,None),
             (None, None)]

   model = np.empty((spectra.shape[0], n_points))
   signal = np.empty((spectra.shape[0], n_points))
   params = np.empty((spectra.shape[0], n_params))
   for ii, xx in enumerate(spectra):
      # We fit to the real spectrum:
      signal[ii] = np.real(xx[idx])
      params[ii] = _do_two_lorentzian_fit(f_ppm[idx], np.real(signal[ii]),
                                      bounds=bounds)
      
      model[ii] = ut.two_lorentzian(f_ppm[idx], *params[ii])
   
   return model, signal, params


def _do_two_lorentzian_fit(freqs, signal, bounds=None):
   """

   Helper function for the Two-Lorentzian fit
   
   """
   # Use the signal for a rough estimate of the parameters for initialization:

   # The local maxima have a zero-crossing in their derivative, so we start by
   # calculating the derivative:
   diff_sig = np.diff(signal)
   # We look for indices that have zero-crossings (in the right direction - we
   # are looking for local maxima, not minima!)
   local_max_idx = []
   for ii in range(len(diff_sig)-1):
      if diff_sig[ii]>0 and diff_sig[ii+1]<0:
        local_max_idx.append(ii)

   # Array-ify it before moving on:
   local_max_idx = np.array(local_max_idx)
   # Our guesses for the location of the interesting local maxima is the two
   # with the largest signals in them: 
   max_idx = np.sort(np.argsort(signal[local_max_idx])[::-1][:2])
   # We sort again, so that we can try to get the first one to be choline:
   max_idx = np.sort(max_idx)
   # And thusly: 
   max_idx_1 = max_idx[0]
   max_idx_2 = max_idx[1]
   # A few of the rest just follow:
   max_sig_1 = signal[max_idx_1]
   max_sig_2 = signal[max_idx_2]
   initial_f0_1 = freqs[max_idx_1]
   initial_f0_2 = freqs[max_idx_2]
   half_max_idx_1 = np.argmin(np.abs(np.real(signal) - max_sig_1/2))
   initial_hwhm_1 = np.abs(initial_f0_1 - freqs[half_max_idx_1])
   half_max_idx_2 = np.argmin(np.abs(np.real(signal) - max_sig_2/2))
   initial_hwhm_2 = np.abs(initial_f0_2 - freqs[half_max_idx_2])
   
   # Everything should be treated as real, except for the phase!
   initial_ph_1 = np.angle(signal[max_idx_1])
   initial_ph_2 = np.angle(signal[max_idx_2])

   # We only fit one offset and one drift, for both functions together! 
   initial_off = np.min(np.real(signal))
   initial_drift = 0

   initial_a_1 = (np.sum(np.real(signal)[max_idx_1:max_idx_1 +
                                         np.abs(half_max_idx_1)*2]) ) * 2

   initial_a_2 = (np.sum(np.real(signal)[max_idx_2:max_idx_2 +
                                         np.abs(half_max_idx_2)*2]) ) * 2

   initial = (initial_f0_1,
              initial_f0_2,
              initial_a_1,
              initial_a_2,
              initial_hwhm_1,
              initial_hwhm_2,
              initial_ph_1,
              initial_ph_2,
              initial_off,
              initial_drift)

   params, _ = lsq.leastsqbound(mopt.err_func, initial,
                                args=(freqs, np.real(signal), ut.two_lorentzian),
                                bounds=bounds)
   return params




def fit_gaussian(spectra, f_ppm, lb=2.6, ub=3.6):
   """
   Fit a gaussian function to the difference spectra to be used for estimation of
   the GABA peak.
   
   Parameters
   ----------
   spectra : array of shape (n_transients, n_points)
      Typically the difference of the on/off spectra in each transient.

   f_ppm : array

   lb, ub: floats
      In ppm, the range over which optimization is bounded
   
   """
   idx = ut.make_idx(f_ppm, lb, ub)
   # We are only going to look at the interval between lb and ub
   n_points = idx.stop - idx.start
   n_params = 5
   fit_func = ut.gaussian
   # Set the bounds for the optimization
   bounds = [(lb,ub), # peak location
             (0,None), # sigma
             (0,None), # amp
             (None, None), # offset
             (None, None)  # rift
             ]

   model = np.empty((spectra.shape[0], n_points))
   signal = np.empty((spectra.shape[0], n_points))
   params = np.empty((spectra.shape[0], n_params))
   for ii, xx in enumerate(spectra):
      # We fit to the real spectrum:
      signal[ii] = np.real(xx[idx])
      # Use the signal for a rough estimate of the parameters for
      # initialization :
      max_idx = np.argmax(signal[ii])
      max_sig = np.max(signal[ii])
      initial_f0 = f_ppm[idx][max_idx]
      half_max_idx = np.argmin(np.abs(signal[ii] - max_sig/2))
      # We estimate sigma as the hwhm:
      initial_sigma = np.abs(initial_f0 - f_ppm[idx][half_max_idx])
      initial_off = np.min(signal[ii])
      initial_drift = 0
      initial_amp = max_sig
      
      initial = (initial_f0,
                 initial_sigma,
                 initial_amp,
                 initial_off,
                 initial_drift)
      
      params[ii], _ = lsq.leastsqbound(mopt.err_func,
                                       initial,
                                       args=(f_ppm[idx],
                                             np.real(signal[ii]),
                                             fit_func), bounds=bounds)

      model[ii] = fit_func(f_ppm[idx], *params[ii])
   
   return model, signal, params


def integrate(func, x, args=(), offset=0, drift=0):
   """
   Integrate a function over the domain x

   Parameters
   ----------
   func : callable
       A function from the domain x to floats. The first input to this function
       has to be x, an array with values to evaluate for, running in monotonic
       order  

   x : float array
      The domain over which to integrate, as sampled. This can be monotonically
      decreasing or monotonically increasing.
      
   args : tuple
       The parameters of func after x.

   offset : 

   Notes
   -----
   We apply the trapezoid rule for integration here, using
   scipy.integrate.trapz.

   See: http://en.wikipedia.org/wiki/Trapezoidal_rule
   
   """
   # If it's monotonically decreasing (as is often the case here), we invert
   # it, so that our results are strictly positive
   if x[1]<x[0]:
      x = x[::-1]
   y = func(x, *args)
   # Correct for offset and drift, if those are present and specified
   # (otherwise default to 0 on both):
   y = y - offset
   y = y - drift * (x-x[0])
   # Use trapezoidal integration on the corrected function: 
   return spi.trapz(y, x)


def simple_auc(spectrum, f_ppm, center=3.00, bandwidth=0.30):
   """
   Calculates area under the curve (no fitting)

   Parameters
   ----------
   spectrum : array of shape (n_transients, n_points)
      Typically the difference of the on/off spectra in each transient.

   center, bandwidth : float
      Determine the limits for the part of the spectrum for which we want
      to calculate the AUC.
      e.g. if center = 3.0, bandwidth = 0.3, lower and upper bounds will be
      2.85 and 3.15 respectively (center +/- bandwidth/2).

   Notes
   -----
   Default center and bandwidth are 3.0 and 0.3ppm respectively
    because of Sanacora 1999 pg 1045:
   "The GABA signal was integrated over a 0.30-ppm bandwidth at 3.00ppm"

   Ref: Sanacora, G., Mason, G. F., Rothman, D. L., Behar, K. L., Hyder, F., Petroff, O. A., ... & Krystal, J. H. (1999). Reduced cortical {gamma}-aminobutyric acid levels in depressed patients determined by proton magnetic resonance spectroscopy. Archives of general psychiatry, 56(11), 1043.

   """
   range = np.max(f_ppm)-np.min(f_ppm)
   dx=float(range)/float(len(f_ppm))
   
   lb = np.floor((np.max(f_ppm)-float(center)+float(bandwidth)/2)/dx)
   ub = np.ceil((np.max(f_ppm)-float(center)-float(bandwidth)/2)/dx)

   auc = trapz(spectrum[ub:lb].real, dx=dx)

   return auc, ub, lb   

