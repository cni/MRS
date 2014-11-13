"""
MRS.analysis
------------

Analysis functions for analysis of MRS data. These include a variety of
functions that can be called independently, or through the interface provided
in :mod:`MRS.api`.

"""
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
import scipy.stats as stats

import MRS.leastsqbound as lsq
import MRS.utils as ut
import MRS.optimize as mopt


def bootstrap_stat(arr, stat=np.mean, n_iters=1000, alpha=0.05):
    """
    Produce a boot-strap distribution of the mean of an array on axis 0

    Parameters
    ---------
    arr : ndarray
       The array with data to be bootstrapped

    stat : callable
        The statistical function to call. will be called as `stat(arr, 0)`, so
        needs to accept that call signature.

    n_iters : int
        The number of bootstrap iterations to sample

    alpha : float
       The confidence interval size will be 1-alpha

    """
    stat_orig = stat(arr, 0)

    boot_arr = np.empty((arr.shape[-1] , n_iters))
    for ii in xrange(n_iters):
        this_arr=arr[np.random.random_integers(0, arr.shape[0]-1, arr.shape[0])]
        boot_arr[:, ii] = stat(this_arr, 0)

    eb = np.array([stats.scoreatpercentile(boot_arr[xx], 1-(alpha/2)) -
                   stats.scoreatpercentile(boot_arr[xx], alpha/2)
                   for xx in range(boot_arr.shape[0])])

    return stat_orig, eb



def separate_signals(data, w_idx=[1, 2, 3]):
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
        
        X = \sum_{i}{w_i S_i}

   Where X is the resulting combined signal, $S_i$ are the individual coil
   signals and $w_i$ are calculated as:

   .. math::
   
        w_i = mean(S_i) / var (S_i)
        
    following [Hall2013]_. In addition, we apply a phase-correction, so that
    all the phases of the signals from each coil are 0

    Parameters
    ----------
    data : float array
       The data as it comes from the scanner, with shape (transients, echos,
    coils, time points) 
    
    w_idx : list
       The indices to the non-water-suppressed transients. Per default we take
        the 2nd-4th transients. We dump the first one, because it seems to be
        quite different than the rest of them...

    coil_dim : int
        The dimension on which the coils are represented. Default: 2

    sampling rate : float
        The sampling rate in Hz. Default : 5000.
  
    References
    ----------

    .. [Hall2013] Emma L. Hall, Mary C. Stephenson, Darren Price, Peter
       G. Morris (2013). Methodology for improved detection of low
       concentration metabolites in MRS: Optimised combination of signals from 
       multi-element coil arrays. Neuroimage 86: 35-42.  

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

    # In each coil, we derive S/(N^2):
    s = np.mean(area_w.reshape(-1, area_w.shape[-1]), 0)
    n = np.var(area_w.reshape(-1, area_w.shape[-1]), 0)
    amp_weight = s/n 
    # Normalize to sum to 1: 
    amp_weight = amp_weight / np.sum(amp_weight)    
    
    # Next, we make sure that all the coils have the same phase. We will use
    # the phase of the Lorentzian to align the phases:
    phase_param = params[..., 3]
    zero_phi_w = np.mean(phase_param.reshape(-1, phase_param.shape[-1]),0)

    # This recalculates the weight with the phase alignment (see page 397 in
    # Wald paper):
    weight = amp_weight * np.exp(-1j * zero_phi_w) 

    # Multiply each one of the signals by its coil-weights and average across
    # coils:
    na = np.newaxis  # Short-hand

    # Collapse across coils for the combination in both the water 
    weighted_w_data = np.mean(np.fft.ifft(np.fft.fftshift(
       weight[na, na, :, na] * fft_w)), coil_dim)
    weighted_w_supp_data = np.mean(np.fft.ifft(np.fft.fftshift(
       weight[na, na, : ,na] * fft_w_supp)) , coil_dim)

    # Normalize each series by the sqrt(rms):
    def normalize_this(x):
       return  x * (x.shape[-1] / (np.sum(np.abs(x))))

    weighted_w_data = normalize_this(weighted_w_data)
    weighted_w_supp_data = normalize_this(weighted_w_supp_data)
    # Squeeze in case that some extraneous dimensions were introduced (can
    # happen for SV data, for example)
    return weighted_w_data.squeeze(), weighted_w_supp_data.squeeze()


def get_spectra(data, filt_method=dict(lb=0.1, filt_order=256),
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
    f : 
         the center frequency of the frequencies represented in the
        spectra

     spectrum_water, spectrum_water_suppressed: 
        The first spectrum is for the data with water not suppressed and
        the second spectrum is for the water-suppressed data.

    Notes
    -----
    This function performs the following operations:

    1. Filtering.
    2. Apodizing/windowing. Optionally, this is done with line-broadening (see
    page 92 of Keeler2005_.
    3. Spectral analysis.
        
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
   Fit a lorentzian function to spectra

   This is used in estimation of the water peak and for estimation of the NAA
   peak.  
   
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
   bounds = [(lb,ub), #peak
             (0,None), #area
             (0,None), #hwhm
             (-np.pi/2, np.pi/2), #phase
             (None,None), #offset
             (None, None)] #drift

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


def _two_func_initializer(freqs, signal):
   """
   This is a helper function for heuristic estimation of the initial parameters
   used in fitting dual peak functions

   _do_two_lorentzian_fit
   _do_two_gaussian_fit
   """
   # Use the signal for a rough estimate of the parameters for initialization:
   r_signal = np.real(signal)
   # The local maxima have a zero-crossing in their derivative, so we start by
   # calculating the derivative:
   diff_sig = np.diff(r_signal)
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
   max_idx = local_max_idx[np.argsort(r_signal[local_max_idx])[::-1][:2]]
   # We sort again, so that we can try to get the first one to be the left peak:
   max_idx = np.sort(max_idx)
   if len(max_idx)==1:
      max_idx = [max_idx[0], max_idx[0]]
   # And thusly: 
   max_idx_1 = max_idx[0]
   max_idx_2 = max_idx[1]
   # A few of the rest just follow:
   max_sig_1 = r_signal[max_idx_1]
   max_sig_2 = r_signal[max_idx_2]
   initial_amp_1 = max_sig_1
   initial_amp_2 = max_sig_2
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

   return (initial_f0_1,
           initial_f0_2,
           initial_amp_1,
           initial_amp_2,
           initial_a_1,
           initial_a_2,
           initial_hwhm_1,
           initial_hwhm_2,
           initial_ph_1,
           initial_ph_2,
           initial_off,
           initial_drift)


def _do_two_lorentzian_fit(freqs, signal, bounds=None):
   """

   Helper function for the Two-Lorentzian fit
   
   """


   initial = _two_func_initializer(freqs, signal)
   # Edit out the ones we want: 
   initial = (initial[0], initial[1],
              initial[4], initial[5],
              initial[6], initial[7],
              initial[8], initial[9],
              initial[10], initial[11])

   # We want to preferntially weight the error on estimating the height of the
   # individual peaks, so we formulate an error-weighting function based on
   # these peaks, which is simply a two-gaussian bumpety-bump:
   w = (ut.gaussian(freqs, initial[0], 0.075, 1, 0, 0) +
        ut.gaussian(freqs, initial[1], 0.075, 1, 0, 0))

   # Further, we want to also optimize on the individual lorentzians error, to
   # restrict the fit space a bit more. For this purpose, we will pass a list
   # of lorentzians with indices into the parameter list, so that we can do
   # that (see mopt.err_func for the mechanics).
   func_list = [[ut.lorentzian, [0,2,4,6,8,9],
                 ut.gaussian(freqs, initial[0], 0.075, 1, 0, 0)],
                [ut.lorentzian, [1,3,5,7,8,9],
                 ut.gaussian(freqs, initial[1], 0.075, 1, 0, 0)]]
   
   params, _ = lsq.leastsqbound(mopt.err_func, initial,
                                args=(freqs, np.real(signal),
                                ut.two_lorentzian, w, func_list),
                                bounds=bounds)
   return params


def _do_two_gaussian_fit(freqs, signal, bounds=None):
   """
   Helper function for the two gaussian fit
   """
   initial = _two_func_initializer(freqs, signal)
   # Edit out the ones we want in the order we want them: 
   initial = (initial[0], initial[1],
              initial[6], initial[7],
              initial[2], initial[3],
              initial[10], initial[11])

   # We want to preferntially weight the error on estimating the height of the
   # individual peaks, so we formulate an error-weighting function based on
   # these peaks, which is simply a two-gaussian bumpety-bump:
   w = (ut.gaussian(freqs, initial[0], 0.075, 1, 0, 0) +
        ut.gaussian(freqs, initial[1], 0.075, 1, 0, 0))

   # Further, we want to also optimize on the individual gaussians error, to
   # restrict the fit space a bit more. For this purpose, we will pass a list
   # of gaussians with indices into the parameter list, so that we can do
   # that (see mopt.err_func for the mechanics).
   func_list = [[ut.gaussian, [0,2,4,6,7],
                 ut.gaussian(freqs, initial[0], 0.075, 1, 0, 0)],
                [ut.gaussian, [1,3,5,6,7],
                 ut.gaussian(freqs, initial[1], 0.075, 1, 0, 0)]]

   params, _ = lsq.leastsqbound(mopt.err_func, initial,
                                args=(freqs, np.real(signal),
                                ut.two_gaussian, w, func_list),
                                bounds=bounds)

   return params



def fit_two_lorentzian(spectra, f_ppm, lb=2.6, ub=3.6):
   """
   Fit a lorentzian function to the sum spectra to be used for estimation of
   the creatine and choline peaks.
   
   Parameters
   ----------
   spectra : array of shape (n_transients, n_points)
      Typically the sum of the on/off spectra in each transient.

   f_ppm : array

   lb, ub : floats
      In ppm, the range over which optimization is bounded
   
   """
   # We are only going to look at the interval between lb and ub
   idx = ut.make_idx(f_ppm, lb, ub)
   n_points = np.abs(idx.stop - idx.start) 
   n_params = 10 # Lotsa params!
   # Set the bounds for the optimization
   bounds = [(lb,ub), #peak1 
             (lb,ub), #peak2 
             (0,None), #area1 
             (0,None), #area2 
             (0,ub-lb), #hwhm1 
             (0,ub-lb), #hwhm2
             (-np.pi/2, np.pi/2), #phase
             (-np.pi/2, np.pi/2), #phase
             (None,None), #offset
             (None, None)] #drift 

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


def fit_two_gaussian(spectra, f_ppm, lb=3.6, ub=3.9):
   """
   Fit a gaussian function to the difference spectra

   This is useful for estimation of the Glx peak, which tends to have two
   peaks.


   Parameters
   ----------
   spectra : array of shape (n_transients, n_points)
      Typically the difference of the on/off spectra in each transient.

   f_ppm : array

   lb, ub : floats
      In ppm, the range over which optimization is bounded

   """
   idx = ut.make_idx(f_ppm, lb, ub)
   # We are only going to look at the interval between lb and ub
   n_points = idx.stop - idx.start
   n_params = 8
   fit_func = ut.two_gaussian
   # Set the bounds for the optimization
   bounds = [(lb,ub), # peak 1 location
             (lb,ub), # peak 2 location
             (0,None), # sigma 1
             (0,None), # sigma 2
             (0,None), # amp 1
             (0,None), # amp 2
             (None, None), # offset
             (None, None),  # drift
             ]

   model = np.empty((spectra.shape[0], n_points))
   signal = np.empty((spectra.shape[0], n_points))
   params = np.empty((spectra.shape[0], n_params))
   for ii, xx in enumerate(spectra):
      # We fit to the real spectrum:
      signal[ii] = np.real(xx[idx])
      params[ii] = _do_two_gaussian_fit(f_ppm[idx], np.real(signal[ii]),
                                      bounds=bounds)

      model[ii] = fit_func(f_ppm[idx], *params[ii])

   return model, signal, params


def _do_scale_fit(freqs, signal, model, w=None):
   """
   Perform a round of fitting to deal with over or under-estimation.
   Scales curve on y-axis but preserves shape.

   Parameters
   ----------
   freqs : array
   signal : array
      The signal that the model is being fit to
   model : array
      The model being scaled
   w : array
      weighting function

   Returns
   -------
   scalefac : array of len(signal)
      the scaling factor for each transient
   scalemodel : array of model.shape
      the scaled model
   """
   scalefac = np.empty(model.shape[0])
   scalemodel = np.empty((model.shape[0], np.real(model).shape[1]))
   scalesignal = np.empty((signal.shape[0], np.real(signal).shape[1]))
   for ii, xx in enumerate(signal): # per transient
      scalesignal[ii] = np.real(xx)
#      ratio = np.empty(scalesignal[ii].shape[0])
#      for ppm, trans in enumerate(scalesignal[ii]):
#          ratio[ppm] = trans/model[ii][ppm]
#      scalefac[ii] = np.mean(ratio,0)
      scalefac[ii] = np.nanmean(scalesignal[ii],0)/np.nanmean(model[ii],0)
      scalemodel[ii] = scalefac[ii] * model[ii]
   return scalefac, scalemodel


def scalemodel(model, scalefac):
   """
   Given a scale factor, multiply by model to get scaled model

   Parameters
   ----------
   model : array
      original model
   scalefac : array of model.shape[0]
      array of scalefactors

   Returns
   -------
   scaledmodel : array
      model scaled by scale factor
   """
   for ii, mm in enumerate(model):
      scaledmodel[ii] = mm * scalefac[ii]
   return scaledmodel


def fit_gaussian(spectra, f_ppm, lb=2.6, ub=3.6):
   """
   Fit a gaussian function to the difference spectra to be used for estimation
   of the GABA peak.
   
   Parameters
   ----------
   spectra : array of shape (n_transients, n_points)
      Typically the difference of the on/off spectra in each transient.

   f_ppm : array

   lb, ub : floats
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
             (None, None)  # drift
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

   Ref: Sanacora, G., Mason, G. F., Rothman, D. L., Behar, K. L., Hyder, F.,
   Petroff, O. A., ... & Krystal, J. H. (1999). Reduced cortical
   {gamma}-aminobutyric acid levels in depressed patients determined by proton
   magnetic resonance spectroscopy. Archives of general psychiatry, 56(11),
   1043.

   """
   range = np.max(f_ppm)-np.min(f_ppm)
   dx=float(range)/float(len(f_ppm))
   
   lb = np.floor((np.max(f_ppm)-float(center)+float(bandwidth)/2)/dx)
   ub = np.ceil((np.max(f_ppm)-float(center)-float(bandwidth)/2)/dx)

   auc = trapz(spectrum[ub:lb].real, dx=dx)

   return auc, ub, lb   



