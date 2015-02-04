"""
MRS.api
-------

Functions and classes for representation and analysis of MRS data. This is the
main module to use when performing routine analysis of MRS data.

"""
import numpy as np
import scipy.stats as stats
import nibabel as nib
import warnings

import MRS.analysis as ana
import MRS.utils as ut
try:
    import MRS.freesurfer as fs
except ImportError:
    warnings.warn("Nipype is not installed. Some functions might not work")
    

class GABA(object):
    """
    Class for analysis of GABA MRS.
    """

    def __init__(self,
                 in_data,
                 w_idx=[1,2,3],
                 line_broadening=5,
                 zerofill=100,
                 filt_method=None,
                 spect_method=dict(NFFT=1024, n_overlap=1023, BW=2),
                 min_ppm=-0.7,
                 max_ppm=4.3,
                 sampling_rate=5000.):
        """
        Parameters
        ----------

        in_data : str
            Path to a nifti file containing MRS data.

        w_idx : list (optional)
            The indices to the non-water-suppressed transients. Per default we
            take the 2nd-4th transients. We dump the first one, because it
            seems to be quite different than the rest of them. 

        line_broadening : float (optional)
           How much to broaden the spectral line-widths (Hz). Default: 5
           
        zerofill : int (optional)
           How many zeros to add to the spectrum for additional spectral
           resolution. Default: 100

        filt_method : dict (optional)
            How/whether to filter the data. Default: None (#nofilter)

        spect_method: dict (optional)
            How to derive spectra. Per default, a simple Fourier transform will
            be derived from apodized time-series, but other methods can also be
            used (see `nitime` documentation for details)
            
        min_ppm, max_ppm : float
           The limits of the spectra that are represented

        sampling_rate : float
           The sampling rate in Hz.
        
        """
        if isinstance(in_data, str):
            # The nifti files follow the strange nifti convention, but we want
            # to use our own logic, which is transients on dim 0 and time on
            # dim -1:
            self.raw_data = np.transpose(nib.load(in_data).get_data(),
                                         [1,2,3,4,5,0]).squeeze()
        elif isinstance(in_data, np.ndarray):
            self.raw_data = in_data

        w_data, w_supp_data = ana.coil_combine(self.raw_data, w_idx=w_idx,
                                               sampling_rate=sampling_rate)
        f_hz, w_supp_spectra = ana.get_spectra(w_supp_data,
                                           line_broadening=line_broadening,
                                           zerofill=zerofill,
                                           filt_method=filt_method,
                                           spect_method=spect_method)

        self.w_supp_spectra = w_supp_spectra

        # Often, there will be some small offset from the on-resonance
        # frequency, which we can correct for. We fit a Lorentzian to each of
        # the spectra from the water-suppressed data, so that we can get a
        # phase-corrected estimate of the frequency shift, instead of just
        # relying on the frequency of the maximum:
        self.w_supp_lorentz = np.zeros(w_supp_spectra.shape[:-1] + (6,))
        for ii in range(self.w_supp_lorentz.shape[0]):
            for jj in range(self.w_supp_lorentz.shape[1]):
                self.w_supp_lorentz[ii,jj]=\
                    ana._do_lorentzian_fit(f_hz, w_supp_spectra[ii,jj])

        # We store the frequency offset for each transient/echo:
        self.freq_offset = self.w_supp_lorentz[..., 0]

        # But for now, we average over all the transients/echos for the
        # correction: 
        mean_freq_offset = np.mean(self.w_supp_lorentz[..., 0])
        f_hz = f_hz - mean_freq_offset
    
        self.water_fid = w_data
        self.w_supp_fid = w_supp_data
        # This is the time-domain signal of interest, combined over coils:
        self.data = ana.subtract_water(w_data, w_supp_data)

        _, spectra = ana.get_spectra(self.data,
                                     line_broadening=line_broadening,
                                     zerofill=zerofill,
                                     filt_method=filt_method)

        self.f_hz = f_hz
        # Convert from Hz to ppm and extract the part you are interested in.
        f_ppm = ut.freq_to_ppm(self.f_hz)
        idx0 = np.argmin(np.abs(f_ppm - min_ppm))
        idx1 = np.argmin(np.abs(f_ppm - max_ppm))
        self.idx = slice(idx1, idx0)
        self.f_ppm = f_ppm
    
        self.echo_off = spectra[:, 1]
        self.echo_on = spectra[:, 0]

        # Calculate sum and difference:
        self.diff_spectra = self.echo_on - self.echo_off
        self.sum_spectra = self.echo_off + self.echo_on

        
    def reset_fits(self):
        """
        This is used to restore the original state of the fits.
        """
        for attr in ['creatine_params', 'creatine_model', 'creatine_signal',
                     'cr_idx', 'creatine_auc', 'gaba_params', 'gaba_model',
                     'gaba_signal', 'gaba_idx', 'gaba_auc', 'glx_params',
                     'glx_model', 'glx_signal', 'glx_idx', 'glx_auc' ]:
            if hasattr(self, attr):
                self.__delattr__(attr)


    def fit_water(self, line_broadening=5, zerofill=100,
                 filt_method=None, min_ppm=-5.0, max_ppm=5.0):
        """

        """
        # Get the water spectrum as well:
        f_hz, w_spectra = ana.get_spectra(self.water_fid,
                                          line_broadening=line_broadening,
                                          zerofill=zerofill,
                                          filt_method=filt_method)

        f_ppm = ut.freq_to_ppm(f_hz)
        # Averaging across echos:
        self.water_spectra = np.mean(w_spectra, 1)
        model, signal, params = ana.fit_lorentzian(self.water_spectra,
                                                   self.f_ppm,
                                                   lb=min_ppm,
                                                   ub=max_ppm)

        # Store the params:
        self.water_model = model
        self.water_signal = signal
        self.water_params = params
        self.water_idx = ut.make_idx(self.f_ppm, min_ppm, max_ppm)
        mean_params = stats.nanmean(params, 0)
        self.water_auc = self._calc_auc(ut.lorentzian, params, self.water_idx)


    def _calc_auc(self, model, params, idx):
        """
        Helper function to calculate the area under the curve of a model

        Parameters
        ----------
        model : callable
            Probably either ut.lorentzian or ut.gaussian, but any function will
            do, as long as its first parameter is an array of frequencies and
            the third parameter controls its amplitude.

        params : ndarray
            Each row of these should contain exactly the number of params that
            the model function expects after the first (frequency)
            parameter. The second column should control the amplitude of the
            function.

        idx :
           Indices to the part of the spectrum over which AUC will be
           calculated.

        """
        # Here's what we are going to do: For each transient, we generate
        # the spectrum for two distinct sets of parameters: one is exactly as
        # fit to the data, the other is the same expect with amplitude set to
        # 0. To calculate AUC, we take the difference between them:
        auc = np.zeros(params.shape[0])
        delta_f = np.abs(self.f_ppm[1]-self.f_ppm[0])
        p = np.copy(params)
        for t in range(auc.shape[0]):
            model1 = model(self.f_ppm[idx], *p[t])
            # This controls the amplitude in both the Gaussian and the
            # Lorentzian: 
            p[t, 1] = 0
            model0 = model(self.f_ppm[idx], *p[t])
            auc[t] = np.sum((model1 - model0) * delta_f)
        return auc

    def _outlier_rejection(self, params, model, signal, ii):
        """
        Helper function to reject outliers

        DRY!
        
        """
        # Z score across repetitions:
        z_score = (params - np.mean(params, 0))/np.std(params, 0)
        # Silence warnings: 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outlier_idx = np.where(np.abs(z_score)>3.0)[0]
            nan_idx = np.where(np.isnan(params))[0]
            outlier_idx = np.unique(np.hstack([nan_idx, outlier_idx]))
            ii[outlier_idx] = 0
            model[outlier_idx] = np.nan
            signal[outlier_idx] = np.nan
            params[outlier_idx] = np.nan

        return model, signal, params, ii


    def fit_creatine(self, reject_outliers=3.0, fit_lb=2.7, fit_ub=3.5):
        """
        Fit a model to the portion of the summed spectra containing the
        creatine and choline signals.

        Parameters
        ----------
        reject_outliers : float or bool
           If set to a float, this is the z score threshold for rejection (on
           any of the parameters). If set to False, no outlier rejection

        fit_lb, fit_ub : float
           What part of the spectrum (in ppm) contains the creatine peak.
           Default (2.7, 3.5)

        Note
        ----
        We use upper and lower bounds that are a variation on the bounds
        mentioned on the GANNET ISMRM2013 poster [1]_.

        [1] RAE Edden et al (2013). Gannet GABA analysis toolkit. ISMRM
        conference poster.

        """
        # We fit a two-lorentz function to this entire chunk of the spectrum,
        # to catch both choline and creatine
        model, signal, params = ana.fit_two_lorentzian(self.sum_spectra,
                                                       self.f_ppm,
                                                       lb=fit_lb,
                                                       ub=fit_ub)

        # Use an array of ones to index everything but the outliers and nans:
        ii = np.ones(signal.shape[0], dtype=bool)
        # Reject outliers:
        if reject_outliers:
            model, signal, params, ii = self._outlier_rejection(params,
                                                                model,
                                                                signal,
                                                                ii)
            
        # We'll keep around a private attribute to tell us which transients
        # were good (this is for both creatine and choline):
        self._cr_transients = np.where(ii)
        
        # Now we separate choline and creatine params from each other (remember
        # that they both share offset and drift!):
        self.choline_params = params[:, (0,2,4,6,8,9)]
        self.creatine_params = params[:, (1,3,5,7,8,9)]
        
        self.cr_idx = ut.make_idx(self.f_ppm, fit_lb, fit_ub)

        # We'll need to generate the model predictions from these parameters,
        # because what we're holding in 'model' is for both together:
        self.choline_model = np.zeros((self.creatine_params.shape[0],
                                    np.abs(self.cr_idx.stop-self.cr_idx.start)))

        self.creatine_model = np.zeros((self.choline_params.shape[0],
                                    np.abs(self.cr_idx.stop-self.cr_idx.start)))
        
        for idx in range(self.creatine_params.shape[0]):
            self.creatine_model[idx] = ut.lorentzian(self.f_ppm[self.cr_idx],
                                                     *self.creatine_params[idx])
            self.choline_model[idx] = ut.lorentzian(self.f_ppm[self.cr_idx],
                                                    *self.choline_params[idx])
        self.creatine_signal = signal
        self.creatine_auc = self._calc_auc(ut.lorentzian,
                                           self.creatine_params,
                                           self.cr_idx)
        self.choline_auc = self._calc_auc(ut.lorentzian,
                                          self.choline_params,
                                          self.cr_idx)


    def _fit_helper(self, fit_spectra, reject_outliers, fit_lb, fit_ub,
                    fit_func):
        """
        This is a helper function for fitting different segments of the spectrum
        with Gaussian functions (GLX and GABA).

        Parameters
        ----------
        fit_spectra : ndarray
           The data to fit

        reject_outliers : float or bool
            Z score for outlier rejection. If set to `False`, not outlier
            rejection.

        fit_lb : float
            The lower bound of the part of the ppm scale for which the Gaussian
            is fit.

        fit_ub : float
            The upper bound of the part of the scale fit.

        fit_func: callable
           e.g. `fit_gaussian`

        Returns
        -------
        choose_transients : tuple
            Indices into the original data's transients dimension to select

        non-outlier transients. If reject_outliers is set to `False`, this is
            all the  transients

        model : ndarray
            The model predicition in each transient, based on the fit.

        signal : ndarray
            The original signal in this part of the difference spectrum.

        params : ndarray
            The Gaussian parameters in each transient as fit.

        this_idx : slice object
            A slice into the part of the spectrum that is fit
        """
        # fit_idx should already be set from fitting the creatine params:
        model, signal, params = fit_func(fit_spectra,
                                         self.f_ppm,
                                         lb=fit_lb,
                                         ub=fit_ub)

        # We'll use these indices to reject outliers (or not):
        ii = np.ones(signal.shape[0], dtype=bool)
        # Reject outliers:
        if reject_outliers:
            model, signal, params, ii = self._outlier_rejection(params,
                                                                model,
                                                                signal,
                                                                ii)

        choose_transients = np.where(ii)
        this_idx = ut.make_idx(self.f_ppm, fit_lb, fit_ub)
        return choose_transients, model, signal, params, this_idx


    def _xval_choose_funcs(self, fit_spectra, reject_outliers, fit_lb, fit_ub,
                           fitters=[ana.fit_gaussian,ana.fit_two_gaussian],
                           funcs = [ut.gaussian, ut.two_gaussian]):
        """ Helper function used to do split-half xvalidation to select among
            alternative models"""

        set1 = fit_spectra[::2]
        set2 = fit_spectra[1::2]

        errs = []
        signal_select = [] 
        # We can loop over functions and try each one out, checking the
        # error in each:
        for fitter in fitters:
            models = []
            signals = []
            for this_set in [set1, set2]:
                choose_transients, model, signal, params, this_idx =\
                    self._fit_helper(this_set, reject_outliers,
                                    fit_lb, fit_ub, fitter)
                models.append(np.nanmean(model[choose_transients], 0))
                signals.append(np.nanmean(signal[choose_transients], 0))

                signal_select.append(signal[choose_transients])
                
            #Cross-validate!
            errs.append(np.mean([ut.rmse(models[0], signals[1]),
                                 ut.rmse(models[1], signals[0])]))
        # We really only need to look at the first two:
        signal_err = ut.rmse(np.nanmean(signal_select[0], 0),
                             np.nanmean(signal_select[1], 0))
        # Based on the errors, choose a function. Also report errors:
        return (fitters[np.argmin(errs)], funcs[np.argmin(errs)], np.min(errs),
                signal_err)
            
    def _xval_model_error(self, fit_spectra, reject_outliers, fit_lb, fit_ub,
                           fitter, func):
        """
        Helper function for calculation of split-half cross-validation model
        error and signal reliability.

        """
        set1 = fit_spectra[::2]
        set2 = fit_spectra[1::2]
        errs = []
        signal_select = [] 
        models = []
        signals = []
        for this_set in [set1, set2]:
            choose_transients, model, signal, params, this_idx =\
                self._fit_helper(this_set, reject_outliers,
                                 fit_lb, fit_ub, fitter)
            models.append(np.nanmean(model[choose_transients], 0))
            signals.append(np.nanmean(signal[choose_transients], 0))

            signal_select.append(signal[choose_transients])
                
        #Cross-validation error estimation:
        model_err = np.mean([ut.rmse(models[0], signals[1]),
                              ut.rmse(models[1], signals[0])])
        # Also for the signal:
        signal_err = ut.rmse(np.nanmean(signal_select[0], 0),
                             np.nanmean(signal_select[1], 0))
        # Based on the errors, choose a function. Also report errors:
        return model_err, signal_err

            
    def fit_gaba(self, reject_outliers=3.0, fit_lb=2.8, fit_ub=3.4,
                 phase_correct=True, fit_func=None):
        """
        Fit either a single Gaussian, or a two-Gaussian to the GABA 3 PPM
        peak.

        Parameters
        ----------
        reject_outliers : float
            Z-score criterion for rejection of outliers, based on their model
            parameter

        fit_lb, fit_ub : float
            Frequency bounds (in ppm) for the region of the spectrum to be
            fit.

        phase_correct : bool
            Where to perform zero-order phase correction based on the fit of
            the creatine peaks in the sum spectra

        fit_func : None or callable (default None).
            If this is set to `False`, an automatic selection will take place,
            choosing between a two-Gaussian and a single Gaussian, based on a
            split-half cross-validation procedure. Otherwise, the requested
            callable function will be fit. Needs to conform to the conventions
            of `fit_gaussian`/`fit_two_gaussian` and
            `ut.gaussian`/`ut.two_gaussian`.

        """
        # We need to fit the creatine, so that we know which transients to
        # exclude in fitting this peak:
        if not hasattr(self, 'creatine_params'):
            self.fit_creatine()

        fit_spectra = np.ones(self.diff_spectra.shape) * np.nan
        # Silence warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_spectra =\
                self.diff_spectra[self._cr_transients].copy()

        if phase_correct:
            for ii, this_spec in enumerate(fit_spectra):
                # Silence warnings:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_spectra[ii] = ut.phase_correct_zero(this_spec,
                            self.creatine_params[self._cr_transients][ii, 3])

        if fit_func is None:
            # Cross-validate!
            fitter, self.gaba_func, self.gaba_model_err, self.gaba_signal_err=\
                self._xval_choose_funcs(fit_spectra,
                                        reject_outliers,
                                        fit_lb, fit_ub)
        # Otherwise, you had better supply a couple of callables that can be
        # used to fit these spectra!
        else:
            fitter = fit_func[0]
            self.gaba_func = fit_func[1]
            self.gaba_model_err, self.gaba_signal_err = \
                self._xval_model_error(fit_spectra, reject_outliers,
                                       fit_lb, fit_ub, fitter, self.gaba_func)
        # Either way, we end up fitting to everything in the end: 
        choose_transients, model, signal, params, this_idx = self._fit_helper(
                                         fit_spectra, reject_outliers,
                                         fit_lb, fit_ub, fitter)

        self._gaba_transients = choose_transients
        self.gaba_model = model
        self.gaba_signal = signal
        self.gaba_params = params
        self.gaba_idx = this_idx
        mean_params = stats.nanmean(params, 0)

        self.gaba_auc =  self._calc_auc(self.gaba_func, params, self.gaba_idx)


    def fit_glx(self, reject_outliers=3.0, fit_lb=3.6, fit_ub=3.9,
                fit_func=None):
        """
        Fit a Gaussian function to the Glu/Gln (GLX) peak at 3.75ppm, +/-
        0.15ppm [Hurd2004]_.  Compare this model to a model
        that treats the Glx signal as two gaussian peaks.  Glx signal
        at. Select between them based on cross-validation

        Parameters
        ----------
        reject_outliers : float or bool
           If set to a float, this is the z score threshold for rejection (on
           any of the parameters). If set to False, no outlier rejection

        fit_lb, fit_ub : float
           What part of the spectrum (in ppm) contains the GLX peak.
           Default (3.5, 4.5)

        scalefit : boolean
           If this is set to true, attempt is made to tighten the fit to the
           peak with a second round of fitting where the fitted curve
           is fit with a scale factor. (default false)

        References
        ----------
        .. [Hurd2004] 2004, Measurement of brain glutamate using TE-averaged
        PRESS at 3T

        """
        # Use everything:
        fit_spectra = self.diff_spectra.copy()

        if fit_func is None:
            # Cross-validate!
            fitter, self.glx_func, self.glx_model_err, self.glx_signal_err=\
                self._xval_choose_funcs(fit_spectra,
                                        reject_outliers,
                                        fit_lb, fit_ub)
        # Otherwise, you had better supply a couple of callables that can be
        # used to fit these spectra!
        else:
            fitter = fit_func[0]
            self.glx_func = fit_func[1]
            self.glx_model_err, self.glx_signal_err = \
                self._xval_model_error(fit_spectra, reject_outliers,
                                       fit_lb, fit_ub, fitter, self.glx_func)

        # Do it!
        choose_transients, model, signal, params, this_idx = self._fit_helper(
                                         fit_spectra, reject_outliers,
                                         fit_lb, fit_ub, fitter)

        self._glx_transients = choose_transients
        self.glx_model = model
        self.glx_signal = signal
        self.glx_params = params
        self.glx_idx = this_idx
        mean_params = stats.nanmean(params, 0)

        self.glx_auc =  self._calc_auc(self.glx_func, params, self.glx_idx)


    def fit_naa(self, reject_outliers=3.0, fit_lb=1.8, fit_ub=2.4,
                 phase_correct=True):
        """
        Fit a Lorentzian function to the NAA peak at ~ 2 ppm.  Example of
        fitting inverted peak: Foerster et al. 2013, An imbalance between
        excitatory and inhibitory neurotransmitters in amyothrophic lateral
        sclerosis revealed by use of 3T proton MRS
        """
        model, signal, params = ana.fit_lorentzian(self.diff_spectra,
                                                   self.f_ppm,
                                                   lb=fit_lb,
                                                   ub=fit_ub)

        # Store the params:
        self.naa_model = model
        self.naa_signal = signal
        self.naa_params = params
        self.naa_idx = ut.make_idx(self.f_ppm, fit_lb, fit_ub)
        mean_params = stats.nanmean(params, 0)
        self.naa_auc = self._calc_auc(ut.lorentzian, params, self.naa_idx)


    def fit_glx2(self, reject_outliers=3.0, fit_lb=3.6, fit_ub=3.9,
                 phase_correct=True, scalefit=False):
        """

        Parameters
        ----------
        reject_outliers : float or bool
           If set to a float, this is the z score threshold for rejection (on
           any of the parameters). If set to False, no outlier rejection

        fit_lb, fit_ub : float
           What part of the spectrum (in ppm) contains the creatine peak.
           Default (3.5, 4.2)

        scalefit : boolean
           If this is set to true, attempt is made to tighten the fit to the
           peak with a second round of fitting where the fitted curve
           is fit with a scale factor. (default false)

        """
        if not hasattr(self, 'creatine_params'):
            self.fit_creatine()

        fit_spectra = self.diff_spectra

        # We fit a two-gaussian function to this entire chunk of the spectrum,
        # to catch both glx peaks
        model, signal, params = ana.fit_two_gaussian(fit_spectra,
                                                     self.f_ppm,
                                                     lb=fit_lb,
                                                     ub=fit_ub)

        # Use an array of ones to index everything but the outliers and nans:
        ii = np.ones(signal.shape[0], dtype=bool)
        # Reject outliers:
        if reject_outliers:
            model, signal, params, ii = self._outlier_rejection(params,
                                                                model,
                                                                signal,
                                                                ii)

        # We'll keep around a private attribute to tell us which transients
        # were good:
        self._glx2_transients = np.where(ii)

        # Now we separate params of the two glx peaks from each other
        # (remember that they both share offset and drift!):
        self.glxp1_params = params[:, (0, 2, 4, 6, 7)]
        self.glxp2_params = params[:, (1, 3, 5, 6, 7)]

        self.glx2_idx = ut.make_idx(self.f_ppm, fit_lb, fit_ub)

        # We'll need to generate the model predictions from these parameters,
        # because what we're holding in 'model' is for both together:
        self.glxp1_model = np.zeros((self.glxp1_params.shape[0],
                                np.abs(self.glx2_idx.stop-self.glx2_idx.start)))

        self.glxp2_model = np.zeros((self.glxp2_params.shape[0],
                                np.abs(self.glx2_idx.stop-self.glx2_idx.start)))

        for idx in range(self.glxp2_params.shape[0]):
            self.glxp2_model[idx] = ut.gaussian(self.f_ppm[self.glx2_idx],
                                                *self.glxp2_params[idx])
            self.glxp1_model[idx] = ut.gaussian(self.f_ppm[self.glx2_idx],
                                                    *self.glxp1_params[idx])

        if scalefit:
            combinedmodel = self.glxp2_model + self.glxp1_model
            scalefac, scalemodel = ana._do_scale_fit(
                self.f_ppm[self.glx2_idx], signal,combinedmodel)
            # Reject outliers:
            scalemodel, signal, params, ii = self._rm_outlier_by_amp(params,
                                                                scalemodel,
                                                                signal,
                                                                ii)
            self.glx2_model = scalemodel
        else:
            self.glx2_model = self.glxp1_model + self.glxp2_model


        self.glx2_signal = signal
        self.glx2_auc = (
            self._calc_auc(ut.gaussian, self.glxp2_params, self.glx2_idx) +
            self._calc_auc(ut.gaussian, self.glxp1_params, self.glx2_idx))


    def _rm_outlier_by_amp(self, params, model, signal, ii):
        """
        Helper function to reject outliers based on mean amplitude
        """
        maxamps = np.nanmax(np.abs(model),0)
        z_score = (maxamps - np.nanmean(maxamps,0))/np.nanstd(maxamps,0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outlier_idx = np.where(np.abs(z_score)>2.0)[0]
            nan_idx = np.where(np.isnan(params))[0]
            outlier_idx = np.unique(np.hstack([nan_idx, outlier_idx]))
            ii[outlier_idx] = 0
            model[outlier_idx] = np.nan
            signal[outlier_idx] = np.nan
            params[outlier_idx] = np.nan

        return model, signal, params, ii


    def est_gaba_conc(self):
        """
        Estimate gaba concentration based on equation adapted from Sanacora
        1999, p1045

        Ref: Sanacora, G., Mason, G. F., Rothman, D. L., Behar, K. L., Hyder,
        F., Petroff, O. A., ... & Krystal, J. H. (1999). Reduced cortical
        $\gamma$-aminobutyric acid levels in depressed patients determined by
        proton magnetic resonance spectroscopy. Archives of general psychiatry,
        56(11), 1043.

        """
        # need gaba_auc and creatine_auc
        if not hasattr(self, 'gaba_params'):
            self.fit_gaba()

        # estimate [GABA] according to equation9
        gaba_conc_est = self.gaba_auc / self.creatine_auc * 1.5 * 9.0
        
        self.gaba_conc_est = gaba_conc_est


    def voxel_seg(self, segfile, MRSfile):
        """
        add voxel segmentation info
        
        Parameters
        ----------
        
        segfile : str
            Path to nifti file with segmentation info (e.g. XXXX_aseg.nii.gz)
        
        MRSfile : str
            Path to MRS nifti file 
        """
        total, grey, white, csf, nongmwm, pGrey, pWhite, pCSF, pNongmwm =\
            fs.MRSvoxelStats(segfile, MRSfile)
        
        self.pGrey = pGrey
        self.pWhite = pWhite
        self.pCSF = pCSF
        self.pNongmwm = pNongmwm




class SingleVoxel(object):
    """
    Class for representation and analysis of single voxel (SV) -PROBE
    experiments.
    """

    def __init__(self, in_file, line_broadening=5, zerofill=100,
                 filt_method=None, min_ppm=-0.7, max_ppm=4.3):
        """
        Parameters
        ----------

        in_file : str
            Path to a nifti file with SV-PROBE MRS data.

        line_broadening : float
           How much to broaden the spectral line-widths (Hz)
           
        zerofill : int
           How many zeros to add to the spectrum for additional spectral
           resolution

        min_ppm, max_ppm : float
           The limits of the spectra that are represented

        fit_lb, fit_ub : float
           The limits for the part of the spectrum for which we fit the
           creatine and GABA peaks. 
        
        """
        self.raw_data = np.transpose(nib.load(in_file).get_data(),
                                     [1,2,3,4,5,0]).squeeze()

        w_data, w_supp_data = ana.coil_combine(self.raw_data, w_idx = range(8),
                                               coil_dim=1)
        # We keep these around for reference, as private attrs
        self._water_data = w_data
        self._w_supp_data = w_supp_data
        # This is the time-domain signal of interest, combined over coils:
        self.data = ana.subtract_water(w_data, w_supp_data)

        f_hz, spectra = ana.get_spectra(self.data,
                                        line_broadening=line_broadening,
                                        zerofill=zerofill,
                                        filt_method=filt_method)
                                           
        self.f_hz = f_hz
        # Convert from Hz to ppm and extract the part you are interested in.
        f_ppm = ut.freq_to_ppm(self.f_hz)
        idx0 = np.argmin(np.abs(f_ppm - min_ppm))
        idx1 = np.argmin(np.abs(f_ppm - max_ppm))
        self.idx = slice(idx1, idx0)
        self.f_ppm = f_ppm
        self.spectra = spectra[:,self.idx]
